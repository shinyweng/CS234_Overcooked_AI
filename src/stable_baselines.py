import gymnasium
import numpy as np
import pygame
import copy
import cv2
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

class Overcooked(gymnasium.Env):

    env_name = "Overcooked-v0"

    # gym checks for the action space and obs space while initializing the env and throws an error if none exists
    # custom_init after __init__ no longer works
    # might as well move all the initilization into the actual __init__
    def __init__(self, base_env, featurize_fn, baselines_reproducible=False):
        """
        base_env: OvercookedEnv
        featurize_fn(mdp, state): fn used to featurize states returned in the 'both_agent_obs' field

        Example creating a gym env:

        mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
        base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
        env = gymnasium.make("Overcooked-v0",base_env = base_env, featurize_fn =base_env.featurize_state_mdp)
        """
        if baselines_reproducible:
            # NOTE:
            # This will cause all agent indices to be chosen in sync across simulation
            # envs (for each update, all envs will have index 0 or index 1).
            # This is to prevent the randomness of choosing agent indexes
            # from leaking when using subprocess-vec-env in baselines (which
            # seeding does not reach) i.e. having different results for different
            # runs with the same seed.
            # The effect of this should be negligible, as all other randomness is
            # controlled by the actual run seeds
            np.random.seed(0)
        self.base_env = base_env
        self.featurize_fn = featurize_fn
        self.observation_space = self._setup_observation_space()
        self.action_space = gymnasium.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.reset()
        self.visualizer = StateVisualizer()

    def _setup_observation_space(self):
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32) * float("inf")
        low = np.zeros(obs_shape, dtype=np.float32)
        return gymnasium.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        assert all(
            self.action_space.contains(a) for a in action
        ), "%r (%s) invalid" % (
            action,
            type(action),
        )
        agent_action, other_agent_action = [
            Action.INDEX_TO_ACTION[a] for a in action
        ]

        if self.agent_idx == 0:
            joint_action = (agent_action, other_agent_action)
        else:
            joint_action = (other_agent_action, agent_action)

        next_state, reward, done, env_info = self.base_env.step(joint_action)
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)

        env_info["policy_agent_idx"] = self.agent_idx

        if "episode" in env_info.keys():
            env_info["episode"]["policy_agent_idx"] = self.agent_idx

        obs = {
            "both_agent_obs": both_agents_ob,
            "overcooked_state": next_state,
            "other_agent_env_idx": 1 - self.agent_idx,
        }
        return obs, reward, done, env_info

    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        self.mdp = self.base_env.mdp
        self.agent_idx = np.random.choice([0, 1])
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)

        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)
        return {
            "both_agent_obs": both_agents_ob,
            "overcooked_state": self.base_env.state,
            "other_agent_env_idx": 1 - self.agent_idx,
        }

    def render(self):
        rewards_dict = {}  # dictionary of details you want rendered in the UI
        for key, value in self.base_env.game_stats.items():
            if key in [
                "cumulative_shaped_rewards_by_agent",
                "cumulative_sparse_rewards_by_agent",
            ]:
                rewards_dict[key] = value

        image = self.visualizer.render_state(
            state=self.base_env.state,
            grid=self.base_env.mdp.terrain_mtx,
            hud_data=StateVisualizer.default_hud_data(
                self.base_env.state, **rewards_dict
            ),
        )

        buffer = pygame.surfarray.array3d(image)
        image = copy.deepcopy(buffer)
        image = np.flip(np.rot90(image, 3), 1)
        image = cv2.resize(image, (2 * 528, 2 * 464))
        return image