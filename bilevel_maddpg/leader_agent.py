import torch
import os
import numpy as np
import torch.nn.functional as F
from bilevel_maddpg.model import Actor, Critic, Cost, Critic_Discrete, Cost_Discrete

# leader agent for constrained stackelberg maddpg
class Leader:
    def __init__(self, args, agent_id): 
        self.args = args
        self.agent_id = agent_id
        self.l_multiplier = args.lagrangian_multiplier
        self.cost_threshold = args.cost_threshold
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, agent_id, 8)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id, 8)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the cost
        self.cost_network = Cost(args)
        self.cost_target_network = Cost(args)
        self.cost_target_network.load_state_dict(self.cost_network.state_dict())
        self.cost_optim = torch.optim.Adam(self.cost_network.parameters(), lr=self.args.lr_critic)

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
        
        if os.path.exists(self.model_path + '/cost_params.pkl'):
            self.cost_network.load_state_dict(torch.load(self.model_path + '/cost_params.pkl'))
            print('Agent {} successfully loaded cost_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/cost_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.cost_target_network.parameters(), self.cost_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, follwer_agent):
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
            u_next_leader = self.actor_target_network(o_next[self.agent_id])
            u_next_follower = follwer_agent.actor_target_network(torch.cat([o_next[follwer_agent.agent_id], u_next_leader], dim=1))
            u_next = [u_next_leader, u_next_follower]

            q_next = self.critic_target_network(o_next[self.agent_id], u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * (1-t.unsqueeze(1)) * q_next).detach()

            c_next = self.cost_target_network(o_next[self.agent_id], u_next).detach()
            target_c = (c.unsqueeze(1) + self.args.gamma * (1-t.unsqueeze(1)) * c_next).detach()

        # the q loss
        q_value = self.critic_network(o[self.agent_id], u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the cost loss
        c_value = self.cost_network(o[self.agent_id], u)
        cost_loss = (target_c - c_value).pow(2).mean()

        # the actor loss
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        u[follwer_agent.agent_id] = follwer_agent.actor_network(torch.cat([o[follwer_agent.agent_id],u[self.agent_id]], dim=1))

        cost_violation = F.relu(self.cost_network(o[self.agent_id], u) - self.cost_threshold)
        actor_loss = (- self.critic_network(o[self.agent_id], u) + self.l_multiplier*cost_violation).mean()
        # actor_loss = - self.critic_network(o[self.agent_id], u).mean()

        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.cost_optim.zero_grad()
        cost_loss.backward(retain_graph=True)
        self.cost_optim.step()

        # update lagrange multiplier
        # self.l_multiplier += self.args.lr_lagrangian*(self.cost_network(o[self.agent_id], u).mean().item()-self.cost_threshold)
        self.l_multiplier += cost_violation.mean().item()*self.args.lr_lagrangian
        self.l_multiplier = max(0,min(self.l_multiplier, self.args.lagrangian_max_bound))

        if self.train_step > 0 and self.train_step % self.args.update_rate == 0:
            self._soft_update_target_network()
            
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1
    
    # select action
    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32)
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
        torch.save(self.cost_network.state_dict(),  self.model_path + '/' + 'cost_params.pkl')

# leader agent for constrained stackelberg q learning
class Leader_Stochastic:
    def __init__(self, args, agent_id):  
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.n_action = self.args.action_dim[agent_id]

        # create the network
        self.critic_network = Critic_Discrete(args)
        self.cost_network = Cost_Discrete(args)

        # build up the target network
        self.critic_target_network = Critic_Discrete(args)
        self.cost_target_network = Cost_Discrete(args)

        # load the weights into the target networks
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.cost_target_network.load_state_dict(self.cost_network.state_dict())

        # create the optimizer
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.cost_optim = torch.optim.Adam(self.cost_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # load the model
        if os.path.exists(self.model_path + '/critic_params.pkl'):
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))
            
        if os.path.exists(self.model_path + '/cost_params.pkl'):
            self.cost_network.load_state_dict(torch.load(self.model_path + '/cost_params.pkl'))
            print('Agent {} successfully loaded cost_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/cost_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.cost_target_network.parameters(), self.cost_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions):
        o = torch.tensor(transitions['o_%d' % self.agent_id], dtype=torch.float32)
        o_next = torch.tensor(transitions['o_next_%d' % self.agent_id], dtype=torch.float32) 
        r = torch.tensor(transitions['r_%d' % self.agent_id], dtype=torch.float32)  
        c = torch.tensor(transitions['c_%d' % self.agent_id], dtype=torch.float32)  
        t = torch.tensor(transitions['t_%d' % self.agent_id], dtype=torch.float32)  
        u, u_next = [], []  
        for agent_id in range(self.args.n_agents):
            u.append(F.one_hot(torch.tensor(transitions['u_%d' % agent_id], dtype=torch.int64).squeeze(), num_classes=self.n_action))
            u_next.append(F.one_hot(torch.tensor(transitions['u_next_%d' % agent_id], dtype=torch.int64).squeeze(), num_classes=self.n_action))
        # print(torch.cat(u,dim=1))
        # calculate the target Q value function
        with torch.no_grad():
            q_next = self.critic_target_network(o_next, u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * (1-t.unsqueeze(1)) * q_next).detach()

            c_next = self.cost_target_network(o_next, u_next).detach()
            target_c = (c.unsqueeze(1) + self.args.gamma * (1-t.unsqueeze(1)) * c_next).detach()

        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the cost loss
        c_value = self.cost_network(o, u)
        cost_loss = (target_c - c_value).pow(2).mean()

        # update the network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.cost_optim.zero_grad()
        cost_loss.backward()
        self.cost_optim.step()

        if self.train_step > 0 and self.train_step % self.args.update_rate == 0:
            self._soft_update_target_network()

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1
    
    # select action
    def select_action(self, o, noise_rate, epsilon, cost_threshold):
        if np.random.uniform() < noise_rate:
            act = np.random.randint(self.n_action)
        else:
            o = torch.tensor(o, dtype=torch.float32)
            act = None
            max_q = float("-inf")

            backup_act = None
            backup_q = float("-inf")
            
            for leader_act in torch.arange(self.n_action):
                for follower_act in torch.arange(self.n_action):
                    u = [F.one_hot(leader_act, num_classes=self.n_action), F.one_hot(follower_act,num_classes=self.n_action)]
                    temp = self.critic_network(o, u, dim=0)
                    if temp>=max_q and self.cost_network(o, u, dim=0)<=cost_threshold:
                    # if temp>=max_q:
                        max_q = temp
                        act = leader_act.item()
                    if temp>=backup_q:
                        backup_q = temp
                        backup_act = leader_act.item()
            # act = act
            if act == None:
                act = backup_act
        return act

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.critic_network.state_dict(),  self.model_path + '/' + 'critic_params.pkl')
        torch.save(self.cost_network.state_dict(),  self.model_path + '/' + 'cost_params.pkl')

