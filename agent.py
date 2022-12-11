import torch
from utils.wrappers import Discrete_actions_space
from gym_derk.envs import DerkEnv
from utils.prova_buffer import ReplayBuffer
from networks.rnn_net import RNNAgent
from networks.qmix_net import Qmix_Net
import random
from copy import copy
import numpy as np
import torch.nn as nn



class Agent_Qmix():

    def __init__(self,env,team):#,replay_buffer):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        
        self.env = env
        self.discrete_actions = Discrete_actions_space(3,3,3)
        self.team = team
        self.n_agents = int(self.env.n_agents / 2)    #teams have the same number of components
        self.team_members_id = self.team_split([i for i in range(6)])
        self.n_actions = self.discrete_actions.count
        self.state_shape = self.env.observation_space.shape[0]
        self.action_dict = self.discrete_actions.actions

        self.RNN_input_shape = self.state_shape + self.discrete_actions.action_len
        self.RNN_hidden_dim = 32
        
        self.rnn_1 = RNNAgent(self.RNN_input_shape, self.RNN_hidden_dim, self.n_actions).to(self.device)
        self.rnn_2 = RNNAgent(self.RNN_input_shape, self.RNN_hidden_dim, self.n_actions).to(self.device)
        self.rnn_3 = RNNAgent(self.RNN_input_shape, self.RNN_hidden_dim, self.n_actions).to(self.device)
        
        self.rnn_agents=[self.rnn_1,self.rnn_2,self.rnn_3]

        self.Qmix_hidden_dim = 32
        self.qmix = Qmix_Net((self.n_agents,self.state_shape), self.n_agents, self.Qmix_hidden_dim).to(self.device)
        self.target_qmix = Qmix_Net((self.n_agents,self.state_shape), self.n_agents, self.Qmix_hidden_dim).to(self.device)
        self.hidden_states=[a.init_hidden().to(self.device) for a in self.rnn_agents]
        #for agent in self.rnn_agents:
        #    self.hidden_states.append(agent.init_hidden())
        
        # Training stuff
        #self.tot_episodes = 500
        #self.max_steps_per_episode = 1000
        self.gamma=0.99
        self.learning_rate=0.00025
        self.epsilon=0.3
        #self.min_epsilon = 0.2
        #self.init_epsilon = self.epsilon
        #self.epsilon_decay = 0.95
        self.batch_size = 2

        self.optimizer = torch.optim.Adam(self.qmix.parameters(),
                                          lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def greedy_action(self,input):
        indices = []
        for i in range(len(self.rnn_agents)):
            res = torch.from_numpy(input[i]).unsqueeze(0).to(self.device)
            idx,_ = self.rnn_agents[i].greedy_action_id(res)
            indices.append(idx)      
        return indices


    def team_split(self,obj):
        if self.team ==1:
            return obj[0:3]
        else:
            return obj[3:6]
    
    def act(self, input, exploit):  
        input = self.team_split(input)
        actions = self.greedy_action(input)
        if not exploit:
            actions = self.discrete_actions.sample()[0]
        return actions
    
    def reset_hidden_states(self):
        for i in range(len(self.rnn_agents)):
            self.hidden_states[i]= self.rnn_agents[i].init_hidden().to(self.device)
        
        

    def update(self,buffer,episode_limit=150):
        #self.load()
        #batch = buffer.sample(self.batch_size)
        qvals = [0,0,0]
        next_qvals = [0,0,0]
        stack_batch_qvals=torch.zeros((self.batch_size,episode_limit,self.n_agents)).to(self.device)#*-9999#(1,episode_limit,agents))
        #(bs,traj,6,64)
        stack_batch_next_qvals=torch.zeros((self.batch_size,episode_limit,self.n_agents)).to(self.device)
        stack_batch_state = torch.zeros(self.batch_size,episode_limit,self.n_agents*2,self.state_shape).to(self.device)
        stack_batch_next_state = torch.zeros(self.batch_size,episode_limit,self.n_agents*2,self.state_shape).to(self.device)
        stack_batch_rewards = torch.zeros(self.batch_size,episode_limit,self.n_agents).to(self.device)
        stack_batch_terminated = torch.zeros(self.batch_size,episode_limit).to(self.device)
        
        batch = buffer.sample(self.batch_size)
        for i in range(self.batch_size):
            self.reset_hidden_states()
            last_action_id = np.ones((3))*4     # 4 is the index of (0,0,0,0,0)
            #stack_t=torch.zeros((1,3))
            traj_len=int(batch['episode_len'][i])+1
            for t in range(traj_len): #trajectory len
                obs = self.team_split(batch['o'][i][t])
                obs_next = self.team_split(batch['o_next'][i][t]) #(3,64)
                rewards = self.team_split(batch['r'][i][t]) #(3,)
                actions_id = self.team_split(batch['a'][i][t]) #(3,1)
                global_state=batch['s'][i][t]
                next_gs = batch['s_next'][i][t]
                terminated = batch['terminated'][i][t]

                stack_batch_state[i,t] = torch.from_numpy(global_state)
                stack_batch_next_state[i,t]=torch.from_numpy(next_gs)
                stack_batch_rewards[i,t]=torch.from_numpy(rewards)
                stack_batch_terminated[i,t]=torch.from_numpy(terminated)
                
                for j in range(len(self.rnn_agents)):      # j for teammate
                    l_a = np.array(self.action_dict[int(last_action_id[j])])
                    input_rnn = np.hstack((obs[j].astype('float32'),l_a.astype('float32')))
                    input_rnn=torch.from_numpy(input_rnn).unsqueeze(0).to(self.device)
                    qvals[j], self.hidden_states[j] = self.rnn_agents[j].forward(input_rnn,self.hidden_states[j])
                    
                    #q_state = torch.rand((32,60,6,64)) #batc_size, traj_len,obs
                
                q_f = torch.cat((qvals[0],qvals[1],qvals[2]),dim=0)
                q_f = torch.gather(q_f,1,torch.LongTensor(actions_id.reshape(-1,1))).squeeze(-1)
        

                stack_batch_qvals[i,t,:]=q_f

                last_action_id = actions_id
           
            stack_batch_next_qvals[i,:-1]=stack_batch_qvals[i,1:]
            if traj_len < episode_limit:
                stack_batch_state[i,traj_len:,:] = torch.from_numpy(global_state)
                stack_batch_next_state[i,traj_len:,:]=torch.from_numpy(next_gs)
                stack_batch_terminated[i,traj_len:] = 1

            for j in range(len(self.rnn_agents)):
                a = np.array(self.action_dict[int(actions_id[j])])
                input_rnn = np.hstack((obs_next[j].astype('float32'),a.astype('float32')))
                input_rnn=torch.from_numpy(input_rnn).unsqueeze(0)
                next_qvals[j], self.hidden_states[j] = self.rnn_agents[j].forward(input_rnn,self.hidden_states[j])
            
            next_q_f = torch.cat((next_qvals[0],next_qvals[1],next_qvals[2]),dim=0)
            next_q_f_max = torch.max(next_q_f,dim =-1)[0]#.reshape(-1,1)
            #next_q_f = torch.gather(next_q_f,1,torch.LongTensor(actions_id.reshape(-1,1))).squeeze(-1)

            stack_batch_next_qvals[i,-1]=next_q_f_max

        #we have all the q vals and the states to feed to qmix
        # in the variables stack_batch_qvals and stack_batch_state

        q_tot = self.qmix.forward(stack_batch_qvals,stack_batch_state).squeeze(-1)
        #print(q_tot.shape) -> (bs,t,1)
        next_qtot_max = self.target_qmix(stack_batch_next_qvals,stack_batch_next_state)
        rews = stack_batch_rewards.sum(-1) 
        target_qtot = rews + (1-stack_batch_terminated)*next_qtot_max.squeeze(-1)*self.gamma
        loss = self.loss_function(q_tot, target_qtot)
        print("the loss is",loss)
        loss.backward()
        self.optimizer.step()
        #self.update_loss.append(loss.item())
        self.save()
        self.update_target_q_net()
    
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
   
    
    def save(self):
        torch.save(self.qmix.state_dict(),  "model_qmix.pt")
        torch.save(self.rnn_1.state_dict(), "model_rnn1.pt")
        torch.save(self.rnn_2.state_dict(), "model_rnn2.pt")
        torch.save(self.rnn_3.state_dict(), "model_rnn3.pt")
    
    def load(self):
        self.qmix.load_state_dict(torch.load("model_qmix.pt", map_location=self.device))
        self.rnn_1.load_state_dict(torch.load("model_rnn1.pt", map_location=self.device))
        self.rnn_2.load_state_dict(torch.load("model_rnn2.pt", map_location=self.device))
        self.rnn_3.load_state_dict(torch.load("model_rnn3.pt", map_location=self.device))
    
    def update_target_q_net(self):
        self.target_qmix.load_state_dict(self.qmix.state_dict())

class Agents():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = DerkEnv(turbo_mode=True)
        self.agent_1 = Agent_Qmix(self.env,team = 1)
        self.agent_2 = Agent_Qmix(self.env,team = 2)
        self.discrete_actions = Discrete_actions_space(3,3,3)
        self.action_dict = self.discrete_actions.actions
        self.action_shape = 5
        self.n_agents = self.env.n_agents
        self.obs_shape = self.env.observation_space.shape[0]
        self.state_shape = self.env.observation_space.shape[0]
        self.last_action = np.zeros((6,5))

        # global vars
        self.observation_n = self.env.reset()
        self.epsilon=0.3


        #buffer 
        self.episode_limit = 60
        self.buffer_size = 50
        self.buffer = ReplayBuffer(self.n_agents,(6, 64),(6, 64),self.buffer_size,self.episode_limit)

        self.episode_batch = {'o': np.zeros([self.episode_limit, self.n_agents, self.obs_shape]),
                        'a': np.zeros([self.episode_limit, self.n_agents]),
                        's': np.zeros([self.episode_limit, self.n_agents,self.state_shape]),  
                        'r': np.zeros([self.episode_limit, self.n_agents]),
                        'o_next': np.zeros([self.episode_limit, self.n_agents, self.obs_shape]),
                        's_next': np.zeros([self.episode_limit, self.n_agents,self.state_shape]),
                        'terminated': np.zeros([self.episode_limit, 1]),
                        'episode_len': 0
                        }

        # simulation
        self.total_rewards = [0,0]



    def make_step(self, observation_n, last_action_n, exploit = [True,True] ):

        old_observation_n = observation_n 
        old_global_state = old_observation_n
        
        input_rnn = np.hstack((observation_n.astype('float32'),last_action_n.astype('float32')))

        action_1_id = self.agent_1.act(input_rnn,exploit[0])
        action_2_id = self.agent_2.act(input_rnn,exploit[1])
        action_n = action_1_id + action_2_id
        
        action_to_do = [ self.action_dict[idx] for idx in action_n]

        observation_n, reward_n, done_n, info = self.env.step(action_to_do)
        
        global_state = observation_n
        #self.total_rewards[0] += sum(reward_n[0:2])
        #self.total_rewards[1] += sum(reward_n[3:5])
        return (old_observation_n, action_n, old_global_state, reward_n,
                     observation_n, global_state, done_n)
    

    def e_choice(self):
        p = random.random()
        if p<self.epsilon:
            return False
        else:
            return True


    #populate the buffer with a full episode
    def roll_in_episode(self,training_ag): # -> with this operation we could do a pre-filling procedure
        #roll in phase
        #training agent is either 0 or 1
        id = 1
        done = False
        self.last_action = np.zeros((6,5))
        

        while (id<self.episode_limit and not done):
            if training_ag:
                e_exploit = [True,self.e_choice()]
            else:
                e_exploit = [self.e_choice(),True]

            (o, a, s, r, o_next, s_next, d)= self.make_step( self.observation_n,self.last_action,e_exploit)
            done = d[0]
            self.observation_n = o_next
            self.episode_batch['o'][id] = o 
            self.episode_batch['a'][id] = a
            self.episode_batch['s'][id] = s 
            self.episode_batch['r'][id] = r 
            self.episode_batch['o_next'][id] = o_next 
            self.episode_batch['s_next'][id] = s_next 
            self.episode_batch['terminated'][id] = d 
            self.episode_batch['episode_len'] = id

            
            id += 1
            #see if it is conveniente
            if done:
                self.observation_n=self.env.reset()
        #store the episode
        self.buffer.store_episode(self.episode_batch)

        
        
        print('Batch saved in the buffer.')
    
    #max_steps must be greater then bs
    def train(self,max_steps=5):
        for ep in range(10):
            for r_i in range(max_steps):
                self.roll_in_episode(0)
                #self.observation_n=self.env.reset()


            self.agent_1.update(self.buffer,self.episode_limit)
        #leva buffer
        #rollin con secondo
        #repeat


    
if __name__ == '__main__':
    agent = Agents()
    agent.roll_in_episode(1)
    sample=agent.buffer.sample(2)
    print('daje brah')
