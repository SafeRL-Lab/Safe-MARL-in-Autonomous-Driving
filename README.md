# Safe MARL in Autonomous Driving

This is a pytorch implementation of Constrained Stackelberg Q-learning(discrete action) and Constrained Stackelberg MADDPG(continuous action). These algorithms are proposed by incorporating the Stackelberg model into Deep Q-learning and MADDPG, and leveraging the Lagrangian multiplier method to deal with the safety constraints. The highway environments used in our experiments are modified from [highway-env](https://github.com/Farama-Foundation/HighwayEnv).

## 1. Installation

``` Bash
# create conda environment
conda create -n env_name python==3.9
conda activate env_name
pip install -r requirements.txt
```

## 2. Quick Start
- create experiment folder, for example, ./merge_env_result/exp2
- define train config in ./merge_env_result/exp2/config.py
- define env config in ./merge_env_result/exp2/env_config.py
- start training by running the following command
- new highway environment not supported yet due to version conflict

```shell
python main_bilevel.py --file-path ./merge_env_result/exp2
```

## 3. Demos
### 3.1 Safe Highway environment
<p align="center">
    <img src="img/highway_env/highway_csmaddpg_1.gif" alt="animated" />
    <img src="img/highway_env/highway_csmaddpg_2.gif" alt="animated" />
</p>

### 3.2 Safe Merge environment
<p align="center">
  <img src="img/merge_env/merge_csq.gif" alt="animated" />
</p>

### 3.3 Safe Roundabout environment
<p align="center">
    <img src="img/roundabout_env/roundabout_csq.gif" alt="animated" />
</p>

### 3.4 Safe Intersection environment
<p align="center">
    <img src="img/intersection_env/intersection_csmaddpg.gif" alt="animated" />
</p>

### 3.5 Safe Racetrack environment
<p align="center">
    <img src="img/racetrack_env/racetrack_csmaddpg.gif" alt="animated" />
</p>





## 4. Results
### 4.1 Safe Highway Environment
| Reward and Training curve                       | 
| ----------------------------------- | 
| ![Alt text](img/highway_env/highway_result.png) | 


### 4.2 Safe Merge Environment
| Leader reward                       | Follower reward                     | Total reward                        |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![Alt text](img/merge_env/leader_reward_merge_sum.png) | ![Alt text](img/merge_env/follower_reward_merge_sum.png) |![Alt text](img/merge_env/total_reward_merge_sum.png)|

| Training curve                       | 
| ----------------------------------- | 
|![Alt text](img/merge_env/crash_merge_sum.png)  |


### 4.3 Safe Roundabout Environment
| Leader reward                       | Follower reward                     | Total reward                        |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![Alt text](img/roundabout_env/leader_reward_roundabout_sum.png) | ![Alt text](img/roundabout_env/follower_reward_roundabout_sum.png) |![Alt text](img/roundabout_env/total_reward_roundabout_sum.png)|

| Training curve                       | 
| ----------------------------------- | 
| ![Alt text](img/roundabout_env/crash_roundabout_sum.png)|

### 4.4 Safe Intersection Environment
| Leader reward                       | Follower reward                     | Total reward                        |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![Alt text](img/intersection_env/leader_reward_intersection_sum.png) | ![Alt text](img/intersection_env/follower_reward_intersection_sum.png) |![Alt text](img/intersection_env/total_reward_intersection_sum.png)|

| Training curve                       | 
| ----------------------------------- | 
|![Alt text](img/intersection_env/crash_intersection_sum.png) |
### 4.5 Safe Racetrack Environment
| Leader reward                       | Follower reward                     | Total reward                        |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![Alt text](img/racetrack_env/leader_reward_racetrack_sum.png) | ![Alt text](img/racetrack_env/follower_reward_racetrack_sum.png) |![Alt text](img/racetrack_env/total_reward_racetrack_sum.png)|

| Training curve                       | 
| ----------------------------------- | 
|![Alt text](img/racetrack_env/crash_racetrack_sum.png) |





