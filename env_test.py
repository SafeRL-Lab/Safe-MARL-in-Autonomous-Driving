import gym
import time
import numpy as np
import highway_env
highway_env.register_highway_envs()

env = gym.make("u-turn-v0")
# # env.configure({ 
# #   "manual_control": True,
# #   "real_time_rendering": True,
# #   "screen_width": 1000,
# #   "screen_height": 1000,
# #   "duration": 20,
# #   "observation": {
# #       "type": "MultiAgentObservation",
# #         "observation_config": {
# #             "type": "Kinematics",
# #             "flatten": True,
# #             "absolute": True,
# #             "see_behind": True,
# #             "normalize": False,
# #             "features": ["x", "y", "vx", "vy"],
# #             "vehicles_count": 2
# #             }
# #   },
# #   "action": {
# #     "type": "MultiAgentAction",
# #         "action_config": {
# #             "type": "DiscreteMetaAction"
# #         }
# #   }
# # })


env.reset()
done = False
while not done:
    act = env.action_space.sample()

    # obs, reward, done, _, _ = env.step(act) 

    # print(env.controlled_vehicles[0].target_speeds)
    # print(env.controlled_vehicles[1].target_speeds)
    # print(".......")
    # done = np.all(done)
    env.render()
    
    # time.sleep(1)

# # import gymnasium as gym

# env = gym.make('u-turn-v0')

# # env.configure({"controlled_vehicles": 2})  # Two controlled vehicles
# # env.configure({"vehicles_count": 1})  # A single other vehicle, for the sake of visualisation
# env.reset(seed=0)

# from matplotlib import pyplot as plt
# # %matplotlib inline
# plt.imshow(env.render())
# plt.title("Controlled vehicles are in green")
# plt.show()