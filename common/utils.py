import numpy as np
import gym
import json
import highway_env
highway_env.register_highway_envs()

def make_highway_env(args):
    env = gym.make(args.scenario_name, render_mode='rgb_array')
    eval_env = gym.make(args.scenario_name, render_mode='rgb_array')

    with open(args.file_path+'/env_config.json','r') as f:
        env.configure(json.load(f))

    with open(args.file_path+'/env_config.json','r') as f:
        eval_env.configure(json.load(f))
    
    env.reset()
    eval_env.reset()

    args.n_players = 2  # agent number
    args.n_agents = 2  # agent number
    args.obs_shape = [8, 8]  # obs dim
    args.action_shape = [1,1] # act dim
    # args.action_shape = [2,2]
    args.action_dim = [5,5] # act num for discrete action
    args.terminal_shape = [1,1] # terminal dim
    args.high_action = 1  # act high for continuous action
    args.low_action = -1  # act low for continuous action

    return env, eval_env, args
    
