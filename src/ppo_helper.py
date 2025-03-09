# ppo_helper.py

# Standard Libraries
import numpy as np 

# Overcooked AI imports 
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent, AgentPair

def get_observation(env, state):
    """
    Get observation (state) from the environment.
    """
    # Get state encodings for player 0 and 1
    encoding = np.array(env.lossless_state_encoding_mdp(state))
    obs0, obs1 = encoding[0], encoding[1]
    
    # Change layout to be (w, h, c) to (c, w, h)
    obs0 = np.transpose(obs0, (2, 0, 1)) 
    obs1 = np.transpose(obs1, (2, 0, 1))
    return obs0, obs1 