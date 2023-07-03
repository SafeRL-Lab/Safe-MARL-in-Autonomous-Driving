# Constrained Stackelberg Q-learning and MADDPG

This is a pytorch implementation of Constrained Stackelberg Q-learning and Constrained Stackelberg MADDPG

## 1. Installation

``` Bash
# create conda environment
conda create -n env_name python==3.9
conda activate env_name
pip install -r requirements.txt
```

## 2. Quick Start

```shell
python main_bilevel.py --file-path ./merge_env_result/exp2
```

## 3. Results
### 3.1 Merge environment
| Leader reward                       | Follower reward                     | Total reward                        |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![Alt text](img/merge_env/leader_reward_merge_sum.png) | ![Alt text](img/merge_env/follower_reward_merge_sum.png) |![Alt text](img/merge_env/total_reward_merge_sum.png)|

| Training curve                       | 
| ----------------------------------- | 
|![Alt text](img/merge_env/crash_merge_sum.png)  |



### 3.2 Roundabout environment
| Leader reward                       | Follower reward                     | Total reward                        |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![Alt text](img/roundabout_env/leader_reward_roundabout_sum.png) | ![Alt text](img/roundabout_env/follower_reward_roundabout_sum.png) |![Alt text](img/roundabout_env/total_reward_roundabout_sum.png)|

| Training curve                       | 
| ----------------------------------- | 
| ![Alt text](img/roundabout_env/crash_roundabout_sum.png)|

### 3.3 Intersection environment
| Leader reward                       | Follower reward                     | Total reward                        |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![Alt text](img/intersection_env/leader_reward_intersection_sum.png) | ![Alt text](img/intersection_env/follower_reward_intersection_sum.png) |![Alt text](img/intersection_env/total_reward_intersection_sum.png)|

| Training curve                       | 
| ----------------------------------- | 
|![Alt text](img/intersection_env/crash_intersection_sum.png) |
### 3.4 Racetrack environment
| Leader reward                       | Follower reward                     | Total reward                        |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![Alt text](img/racetrack_env/leader_reward_racetrack_sum.png) | ![Alt text](img/racetrack_env/follower_reward_racetrack_sum.png) |![Alt text](img/racetrack_env/total_reward_racetrack_sum.png)|

| Training curve                       | 
| ----------------------------------- | 
|![Alt text](img/racetrack_env/crash_racetrack_sum.png) |

## 4. Demos
### 4.1 Merge environment
![Alt text](img/merge_env/merge_csq.gif)
### 4.2 Roundabout environment
![Alt text](img/roundabout_env/roundabout_csq.gif)
### 4.3 Intersection environment
![Alt text](img/intersection_env/intersection_csmaddpg.gif)
### 4.4 Racetrack environment
![Alt text](img/racetrack_env/racetrack_csmaddpg.gif)
## Note


