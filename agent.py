import torch
from utils.wrappers import Discrete_actions_space
from gym_derk.envs import DerkEnv
from utils.prova_buffer import ReplayBuffer
from networks.rnn_net import RNNAgent
from networks.qmix_net import Qmix_Net
import random

import numpy as np



class Agent_Qmix():

    def __init__(self,env,team):#,replay_buffer):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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
        
        self.rnn_1 = RNNAgent(self.RNN_input_shape, self.RNN_hidden_dim, self.n_actions)
        self.rnn_2 = RNNAgent(self.RNN_input_shape, self.RNN_hidden_dim, self.n_actions)
        self.rnn_3 = RNNAgent(self.RNN_input_shape, self.RNN_hidden_dim, self.n_actions)
        
        self.rnn_agents=[self.rnn_1,self.rnn_2,self.rnn_3]

        self.Qmix_hidden_dim = 32
        self.qmix = Qmix_Net((self.n_agents,self.state_shape), self.n_agents, self.Qmix_hidden_dim)
        
        self.hidden_states = [self.rnn_1.init_hidden(),self.rnn_2.init_hidden(),self.rnn_3.init_hidden()]
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

    def greedy_action(self,input):
        indices = []
        for i in range(len(self.rnn_agents)):
            idx,_ = self.rnn_agents[i].greedy_action_id(torch.from_numpy(input[i]).unsqueeze(0))
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
        self.hidden_states[0] = self.rnn_1.init_hidden()
        self.hidden_states[1] = self.rnn_2.init_hidden()
        self.hidden_states[2] = self.rnn_3.init_hidden()
        

    def update(self,buffer):
        #self.load()
        #batch = buffer.sample(self.batch_size)
        qvals = [0,0,0]
        batch = buffer.sample(self.batch_size)
        for i in range(self.batch_size):
            self.reset_hidden_states()
            last_action_id = np.ones((3))*4     # 4 is the index of (0,0,0,0,0)
            for t in range(int(batch['episode_len'][i])): #trajectory len
                obs = self.team_split(batch['o'][i][t])
                obs_next = self.team_split(batch['o_next'][i][t]) #(3,64)
                rewards = self.team_split(batch['r'][i][t]) #(3,)
                actions_id = self.team_split(batch['a'][i][t]) #(3,1)
                for j in range(len(self.rnn_agents)):      # j for teammate
                    #print(last_action_id[j])
                    l_a = np.array(self.action_dict[int(last_action_id[j])])
                    input_rnn = np.hstack((obs[j].astype('float32'),l_a.astype('float32')))
                    input_rnn=torch.from_numpy(input_rnn).unsqueeze(0)
                    #fare lista qvals
                    qvals[j], self.hidden_states[j] = self.rnn_agents[j].forward(input_rnn,self.hidden_states[j])
                    #print(qvals)
                last_action_id = actions_id
                q_f = torch.cat((qvals[0],qvals[1],qvals[2]),dim=0)
                q_f = torch.gather(q_f,1,torch.LongTensor(actions_id.reshape(-1,1)))
                print("the shape of qvals is",q_f.shape)
                
                #q_vals = torch.rand(32,60,3) #batchsize, traj_len, q_vals
                
                #cat(q1,q2...qn)
                #compute q -> TD_LOSS




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
        self.episode_limit = 20
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
        #store the episode
        self.buffer.store_episode(self.episode_batch)
        

        print('Batch saved in the buffer.')
    
    
    def train(self,max_steps=2):
        #while smth
        for _ in range(max_steps):
            self.roll_in_episode(0)
            print("finished_roll")
            self.observation_n=self.env.reset()

        
        self.agent_1.update(self.buffer)
        #leva buffer
        #rollin con secondo
        #repeat


    
if __name__ == '__main__':
    agent = Agents()
    agent.roll_in_episode(1)
    sample=agent.buffer.sample(2)
    print('daje brah')
