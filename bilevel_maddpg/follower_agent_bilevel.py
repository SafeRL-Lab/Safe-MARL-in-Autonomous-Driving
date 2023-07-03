import torch
import os
import numpy as np
import torch.nn.functional as F
from bilevel_maddpg.model import Actor, Critic, Cost, Critic_Discrete, Cost_Discrete

# follower agent for unconstrained stackelberg maddpg
class Follower_Bilevel:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, agent_id, 9)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id, 9)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # load model
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            
        if os.path.exists(self.model_path + '/critic_params.pkl'):
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, leader_agent):
        r = torch.tensor(transitions['r_%d' % self.agent_id], dtype=torch.float32)  
        c = torch.tensor(transitions['c_%d' % self.agent_id], dtype=torch.float32)
        t = torch.tensor(transitions['t_%d' % self.agent_id], dtype=torch.float32)  
        o, u, o_next = [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(torch.tensor(transitions['o_%d' % agent_id], dtype=torch.float32))
            u.append(torch.tensor(transitions['u_%d' % agent_id], dtype=torch.float32))
            o_next.append(torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float32))

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            u_next_leader = leader_agent.actor_target_network(o_next[leader_agent.agent_id])
            u_next_follower = self.actor_target_network(torch.cat([o_next[self.agent_id], u_next_leader], dim=1))
            u_next = [u_next_leader, u_next_follower]

            q_next = self.critic_target_network(o_next[self.agent_id], u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * (1-t.unsqueeze(1)) * q_next).detach()

        # the q loss
        q_value = self.critic_network(o[self.agent_id], u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        u[leader_agent.agent_id] = leader_agent.actor_network(o[leader_agent.agent_id])
        u[self.agent_id] = self.actor_network(torch.cat([o[self.agent_id],u[leader_agent.agent_id]], dim=1))
        actor_loss = - self.critic_network(o[self.agent_id], u).mean()

        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        if self.train_step > 0 and self.train_step % self.args.update_rate == 0:
            self._soft_update_target_network()

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1
    
    # select action
    def select_action(self, o, leader_action, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32)
            leader_action = torch.tensor(leader_action, dtype=torch.float32)
            inputs = torch.cat([inputs, leader_action])
            pi = self.actor_network(inputs)
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.actor_network.state_dict(), self.model_path + '/' + 'actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  self.model_path + '/' + 'critic_params.pkl')


