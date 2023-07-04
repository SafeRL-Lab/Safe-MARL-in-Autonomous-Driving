from tqdm import tqdm
from bilevel_maddpg.replay_buffer import Buffer
from bilevel_maddpg.leader_agent import Leader, Leader_Stochastic
from bilevel_maddpg.follower_agent import Follower, Follower_Stochastic
from bilevel_maddpg.leader_agent_bilevel import Leader_Bilevel
from bilevel_maddpg.follower_agent_bilevel import Follower_Bilevel
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# implemntation of Constrained Bilevel RL algorithm
class Runner_C_Bilevel:
    def __init__(self, args, env, eval_env=None):
        self.args = args
        # init noise and epsilon
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        # set min noise and epsilon
        self.min_noise = args.min_noise_rate
        self.min_epsilon = args.min_epsilon
        # set max episode len
        self.episode_limit = args.max_episode_len
        # train env and eval env
        self.env = env
        self.eval_env = eval_env
        # replay buffer
        self.buffer = Buffer(args)
        # save path for training results
        self.save_path = self.args.save_dir
        # reward or cost record
        self.reward_record = [[] for i in range(args.n_agents)]
        self.arrive_record = []
        self.leader_arrive_record = []
        self.follower_arrive_record = []
        self.crash_record = []
        # init agents
        self._init_agents()
        # make dir
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        self.leader_agent = Leader(self.args, 0)
        self.follower_agent = Follower(self.args, 1)

    def run(self):
        returns = []
        total_reward = [0, 0]
        done = [False*self.args.n_agents]
        info = None
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step==0 or np.all(done):
                # save episode rewards
                for i in range(self.args.n_agents):
                    self.reward_record[i].append(total_reward[i])
                # save first arrive record
                if info is not None:
                    if self.args.scenario_name == "racetrack-v0":
                        self.crash_record.append(info["crashed"])
                    else:
                        if info["first_arrived"]==1:
                            self.leader_arrive_record.append(1)
                            self.follower_arrive_record.append(0)
                        elif info["first_arrived"]==2:
                            self.leader_arrive_record.append(0)
                            self.follower_arrive_record.append(1)
                        else:
                            self.leader_arrive_record.append(0)
                            self.follower_arrive_record.append(0)
                # reset episode total reward
                total_reward = [0, 0]
                # reset
                s, info = self.env.reset()
                # reshape observation
                s = np.array(s).reshape((2, 8))
            with torch.no_grad():
                # choose actions
                leader_action = self.leader_agent.select_action(s[0], self.noise, self.epsilon)
                follower_action = self.follower_agent.select_action(s[1], leader_action, self.noise, self.epsilon)

            u = [leader_action, follower_action]
            actions = (leader_action, follower_action)
            # step simulation
            s_next, r, done, truncated_n, info = self.env.step(actions)
            # reshape observation
            s_next = np.array(s_next).reshape((2, 8))
            # cost
            c = info["cost"]
            # next actions(place holder, not used)
            u_next = [0, 0]
            # store transitions
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents], u_next, done[:self.args.n_agents], c=c)
            # observe update
            s = s_next
            # accumulate episode reward
            for i in range(self.args.n_agents):
                total_reward[i]+=r[i]
            # train
            if self.buffer.current_size >= self.args.sample_size:
                transitions = self.buffer.sample(self.args.batch_size)
                self.leader_agent.train(transitions, self.follower_agent)
                self.follower_agent.train(transitions, self.leader_agent)
            # plot reward
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                np.save(self.save_path + '/reward_record.npy', self.reward_record)
                np.save(self.save_path + '/leader_arrive_record.npy', self.leader_arrive_record)
                np.save(self.save_path + '/follower_arrive_record.npy', self.follower_arrive_record)
                np.save(self.save_path + '/crash_record.npy', self.crash_record)
                self.evaluate()
            self.noise = max(self.min_noise, self.noise - 0.0000005)
            self.epsilon = max(self.min_epsilon, self.epsilon - 0.0000005)
            
        # save data
        np.save(self.save_path + '/reward_record.npy', self.reward_record)
        np.save(self.save_path + '/leader_arrive_record.npy', self.leader_arrive_record)
        np.save(self.save_path + '/follower_arrive_record.npy', self.follower_arrive_record)
        np.save(self.save_path + '/crash_record.npy', self.crash_record)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s, info = self.eval_env.reset()
            s = np.array(s).reshape((2, 8))
            rewards = [0, 0]
            for time_step in range(self.args.evaluate_episode_len):
                self.eval_env.render()
                with torch.no_grad():
                    leader_action = self.leader_agent.select_action(s[0], self.noise, self.epsilon)
                    follower_action = self.follower_agent.select_action(s[1], leader_action, self.noise, self.epsilon)
                actions = tuple([leader_action, follower_action])
                s_next, r, done, truncated_n, info = self.eval_env.step(actions)
                s_next = np.array(s_next).reshape((2, 8))
                rewards[0] += r[0]
                rewards[1] += r[1]
                s = s_next
                # time.sleep(1)
                if np.all(done):
                    print('crash', info["crashed"])
                    break
            returns.append(rewards)
            print('Returns is', rewards)
        return np.sum(returns, axis=0) / self.args.evaluate_episodes
    
    def record_video(self):
        s, info = self.eval_env.reset()
        video_recorder = VideoRecorder(self.eval_env, path=self.args.save_dir+'/video.mp4', enabled=True)

        s = np.array(s).reshape((2, 8))
        for time_step in range(self.args.evaluate_episode_len):
            self.eval_env.render()
            video_recorder.capture_frame()
            with torch.no_grad():
                leader_action = self.leader_agent.select_action(s[0], 0, 0)
                follower_action = self.follower_agent.select_action(s[1], leader_action, 0, 0)
            actions = tuple([leader_action, follower_action])
            s_next, r, done, truncated_n, info = self.eval_env.step(actions)
            s_next = np.array(s_next).reshape((2, 8))
            s = s_next
            if np.all(done):
                break
        video_recorder.close()


# implemntation of Bilevel RL algorithm
class Runner_Bilevel:
    def __init__(self, args, env, eval_env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.min_noise = args.min_noise_rate
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.eval_env = eval_env
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir
        self.reward_record = [[] for i in range(args.n_agents)]
        self.arrive_record = []
        self.leader_arrive_record = []
        self.follower_arrive_record = []
        self.crash_record = []
        self._init_agents()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        self.leader_agent = Leader_Bilevel(self.args, 0)
        self.follower_agent = Follower_Bilevel(self.args, 1)

    def run(self):
        returns = []
        total_reward = [0, 0]
        done = [False*self.args.n_agents]
        info = None
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step==0 or np.all(done):
                # save episode rewards
                for i in range(self.args.n_agents):
                    self.reward_record[i].append(total_reward[i])
                # save first arrive record
                if info is not None:
                    if info is not None:
                        if self.args.scenario_name == "racetrack-v0":
                            self.crash_record.append(info["crashed"])
                        else:
                            if info["first_arrived"]==1:
                                self.leader_arrive_record.append(1)
                                self.follower_arrive_record.append(0)
                            elif info["first_arrived"]==2:
                                self.leader_arrive_record.append(0)
                                self.follower_arrive_record.append(1)
                            else:
                                self.leader_arrive_record.append(0)
                                self.follower_arrive_record.append(0)
                # reset episode total reward
                total_reward = [0, 0]
                # reset
                s, info = self.env.reset()
                # reshape observation
                s = np.array(s).reshape((2, 8))
            with torch.no_grad():
                # choose actions
                leader_action = self.leader_agent.select_action(s[0], self.noise, self.epsilon)
                follower_action = self.follower_agent.select_action(s[1], leader_action, self.noise, self.epsilon)

            u = [leader_action, follower_action]
            actions = (leader_action, follower_action)
            # step simulation
            s_next, r, done, truncated_n, info = self.env.step(actions)
            # reshape observation
            s_next = np.array(s_next).reshape((2, 8))
            # cost
            c = info["cost"]
            # next actions(place holder, not used)
            u_next = [0, 0]
            # store transitions
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents], u_next, done[:self.args.n_agents], c=c)
            # observe update
            s = s_next
            # accumulate episode reward
            for i in range(self.args.n_agents):
                total_reward[i]+=r[i]
            # train
            if self.buffer.current_size >= self.args.sample_size:
                transitions = self.buffer.sample(self.args.batch_size)
                self.leader_agent.train(transitions, self.follower_agent)
                self.follower_agent.train(transitions, self.leader_agent)
            # plot reward
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                np.save(self.save_path + '/reward_record.npy', self.reward_record)
                np.save(self.save_path + '/leader_arrive_record.npy', self.leader_arrive_record)
                np.save(self.save_path + '/follower_arrive_record.npy', self.follower_arrive_record)
                np.save(self.save_path + '/crash_record.npy', self.crash_record)
            self.noise = max(self.min_noise, self.noise - 0.0000005)
            self.epsilon = max(self.min_epsilon, self.epsilon - 0.0000005)
        # save data
        np.save(self.save_path + '/reward_record.npy', self.reward_record)
        np.save(self.save_path + '/leader_arrive_record.npy', self.leader_arrive_record)
        np.save(self.save_path + '/follower_arrive_record.npy', self.follower_arrive_record)
        np.save(self.save_path + '/crash_record.npy', self.crash_record)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s, info = self.eval_env.reset()
            s = np.array(s).reshape((2, 8))
            rewards = [0, 0]
            for time_step in range(self.args.evaluate_episode_len):
                self.eval_env.render()
                with torch.no_grad():
                    leader_action = self.leader_agent.select_action(s[0], self.noise, self.epsilon)
                    follower_action = self.follower_agent.select_action(s[1], leader_action, self.noise, self.epsilon)
                actions = tuple([leader_action, follower_action])
                s_next, r, done, truncated_n, info = self.eval_env.step(actions)
                s_next = np.array(s_next).reshape((2, 8))
                rewards[0] += r[0]
                rewards[1] += r[1]
                s = s_next
                # time.sleep(1)
                if np.all(done):
                    break
            returns.append(rewards)
            print('Returns is', rewards)
        return np.sum(returns, axis=0) / self.args.evaluate_episodes

    def record_video(self):
        s, info = self.eval_env.reset()
        video_recorder = VideoRecorder(self.eval_env, path=self.args.save_dir+'/video.mp4', enabled=True)

        s = np.array(s).reshape((2, 8))
        for time_step in range(self.args.evaluate_episode_len):
            self.eval_env.render()
            video_recorder.capture_frame()
            with torch.no_grad():
                leader_action = self.leader_agent.select_action(s[0], 0, 0)
                follower_action = self.follower_agent.select_action(s[1], leader_action, 0, 0)
            actions = tuple([leader_action, follower_action])
            s_next, r, done, truncated_n, info = self.eval_env.step(actions)
            s_next = np.array(s_next).reshape((2, 8))
            s = s_next
            if np.all(done):
                break
        video_recorder.close()

    def analysis(self, agent_id):
        plt.figure()
        plt.plot(self.reward_record[agent_id])
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        plt.savefig(self.save_path + '/reward_agent{}.png'.format(agent_id), format='png')
    
# implementation of constrained Bilevel RL algorithm for discrete action space
class Runner_Stochastic:
    def __init__(self, args, env, eval_env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.min_noise = args.min_noise_rate
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.eval_env = eval_env
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir
        self.reward_record = [[] for i in range(args.n_agents)]
        self.leader_arrive_record = []
        self.follower_arrive_record = []
        self.crash_record = []
        self.n_action = 5
        self.enable_cost = args.enable_cost
        self.cost_threshold = args.cost_threshold
        self._init_agents()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        self.leader_agent = Leader_Stochastic(self.args, 0)
        self.follower_agent = Follower_Stochastic(self.args, 1)

    def run(self):
        returns = []
        total_reward = [0, 0]
        done = [False*self.args.n_agents]
        info = None
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step==0 or np.all(done):
                # save episode rewards
                for i in range(self.args.n_agents):
                    self.reward_record[i].append(total_reward[i])
                # save first arrive record
                if info is not None:
                    self.crash_record.append(info["crash"])
                    self.leader_arrive_record.append(info["leader_arrived"])
                    self.follower_arrive_record.append(info["follower_arrived"])
                # reset episode total reward
                total_reward = [0, 0]
                # reset
                s, info = self.env.reset()
                # reshape observation
                s = np.array(s).reshape((2, 8))
            with torch.no_grad():
                # choose actions
                leader_action = self.leader_agent.select_action(s[0], self.noise, self.epsilon, self.cost_threshold)
                follower_action = self.follower_agent.select_action(s[1], leader_action, self.noise, self.epsilon, self.cost_threshold)
            u = [leader_action, follower_action]
            actions = (leader_action, follower_action)
            # step simulation
            s_next, r, done, truncated_n, info = self.env.step(actions)
            # reshape observation
            s_next = np.array(s_next).reshape((2, 8))
            # cost
            c = info["cost"]
            # next actions(place holder, not used)
            u_next = [0,0]
            # store transitions
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents], u_next, done[:self.args.n_agents], c=c)
            # observe update
            s = s_next
            # accumulate episode reward
            for i in range(self.args.n_agents):
                total_reward[i]+=r[i]
            # train
            if self.buffer.current_size >= self.args.sample_size:
                # sample
                transitions = self.buffer.sample(self.args.batch_size)
                # add next actions
                transitions = self.add_target_action(transitions)
                # train
                self.leader_agent.train(transitions)
                self.follower_agent.train(transitions)
            # plot reward
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                np.save(self.save_path + '/reward_record.npy', self.reward_record)
                np.save(self.save_path + '/leader_arrive_record.npy', self.leader_arrive_record)
                np.save(self.save_path + '/follower_arrive_record.npy', self.follower_arrive_record)
                np.save(self.save_path + '/crash_record.npy', self.crash_record)
            self.noise = max(self.min_noise, self.noise - 0.000005)
            self.epsilon = max(self.min_epsilon, self.epsilon - 0.000005)
        
        # save data
        np.save(self.save_path + '/reward_record.npy', self.reward_record)
        np.save(self.save_path + '/leader_arrive_record.npy', self.leader_arrive_record)
        np.save(self.save_path + '/follower_arrive_record.npy', self.follower_arrive_record)
        np.save(self.save_path + '/crash_record.npy', self.crash_record)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s, info = self.eval_env.reset()
            s = np.array(s).reshape((2, 8))
            rewards = [0, 0]
            for time_step in range(self.args.evaluate_episode_len):
                self.eval_env.render()
                with torch.no_grad():
                    leader_action = self.leader_agent.select_action(s[0], self.noise, self.epsilon, self.cost_threshold)
                    follower_action = self.follower_agent.select_action(s[1], leader_action, self.noise, self.epsilon, self.cost_threshold)
                actions = (leader_action, follower_action)
                s_next, r, done, truncated_n, info = self.eval_env.step(actions)
                s_next = np.array(s_next).reshape((2, 8))
                rewards[0] += r[0]
                rewards[1] += r[1]
                s = s_next
                if np.all(done):
                    # print(info)
                    break
            returns.append(rewards)
            print('Returns is', rewards)
        return np.sum(returns, axis=0) / self.args.evaluate_episodes
    
    def add_target_action(self, transitions):
        leader_obs = (torch.tensor(transitions['o_%d' % 0], dtype=torch.float32))
        follower_obs = (torch.tensor(transitions['o_%d' % 1], dtype=torch.float32))
        for i in range(self.args.batch_size):
            next_leader_act = self.leader_agent.select_action(leader_obs[i], self.noise, self.epsilon, self.cost_threshold)
            next_follower_act = self.follower_agent.select_action(follower_obs[i], next_leader_act, self.noise, self.epsilon, self.cost_threshold)
            
            transitions['u_next_%d' % 0][i] = next_leader_act
            transitions['u_next_%d' % 1][i] = next_follower_act
        return transitions

    def record_video(self):
        s, info = self.eval_env.reset()
        video_recorder = VideoRecorder(self.eval_env, path=self.args.save_dir+'/video.mp4', enabled=True)

        s = np.array(s).reshape((2, 8))
        for time_step in range(self.args.evaluate_episode_len):
            self.eval_env.render()
            video_recorder.capture_frame()
            with torch.no_grad():
                leader_action = self.leader_agent.select_action(s[0], self.noise, self.epsilon, self.cost_threshold)
                follower_action = self.follower_agent.select_action(s[1], leader_action, self.noise, self.epsilon, self.cost_threshold)
            actions = tuple([leader_action, follower_action])
            s_next, r, done, truncated_n, info = self.eval_env.step(actions)
            s_next = np.array(s_next).reshape((2, 8))
            s = s_next
            if np.all(done):
                break
        video_recorder.close()

