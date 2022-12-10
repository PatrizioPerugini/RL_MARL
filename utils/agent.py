import torch
from wrappers import Discrete_actions_space
from gym_derk.envs import DerkEnv
from prova_buffer import ReplayBuffer

import numpy as np



class Agent_Qmix():

    def __init__(self,env,team):#,replay_buffer):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.discrete_actions = Discrete_actions_space(3,3,3)
        self.team = team
        self.n_agents = env.n_agents / 2    #teams have the same number of components
        self.team_members_id = self.team_split([i for i in range(6)])
        
        # Training stuff
        #self.tot_episodes = 500
        #self.max_steps_per_episode = 1000
        #self.gamma=0.99
        #self.learning_rate=0.00025
        #self.epsilon=0.7
        #self.min_epsilon = 0.2
        #self.init_epsilon = self.epsilon
        #self.epsilon_decay = 0.95
        #self.batch_size = 64      

        

    def team_split(self,obj):
        if self.team ==1:
            return obj[0:3]
        else:
            return obj[3:6]

    
    def act(self, observation_n, mode = 'explore'):  
        obs = self.team_split(observation_n)
        if mode == 'exploit':
          actions = self.greedy_action(obs)
        else:
          actions = [ self.discrete_actions.sample() for _ in self.team_members_id]
        return actions
    

class Agents():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = DerkEnv()
        self.agent_1 = Agent_Qmix(self.env,team = 1)
        self.agent_2 = Agent_Qmix(self.env,team = 2)
        self.discrete_actions = Discrete_actions_space(3,3,3)
        self.n_actions = 5
        self.n_agents = self.env.n_agents
        self.obs_shape = self.env.observation_space.shape[0]
        self.state_shape = self.env.observation_space.shape[0]

        # global vars
        self.observation_n = self.env.reset()

        #buffer 
        self.episode_limit = 60
        self.buffer_size = 50
        print(self.obs_shape)
        self.buffer = ReplayBuffer(self.n_actions,self.n_agents,(6, 64),
                                    (6, 64),self.buffer_size,self.episode_limit)

        self.episode_batch = {'o': np.zeros([self.episode_limit, self.n_agents, self.obs_shape]),
                        'a': np.zeros([self.episode_limit, self.n_agents, self.n_actions]),
                        's': np.zeros([self.episode_limit, self.n_agents,self.state_shape]),  
                        'r': np.zeros([self.episode_limit, self.n_agents]),
                        'o_next': np.zeros([self.episode_limit, self.n_agents, self.obs_shape]),
                        's_next': np.zeros([self.episode_limit, self.n_agents,self.state_shape]),
                        'terminated': np.zeros([self.episode_limit, 1]),
                        'episode_len': 0
                        }


        # simulation
        self.total_rewards = [0,0]



    def make_step(self, observation_n):
        old_observation_n = observation_n 
        old_global_state = old_observation_n
        mode1 = 'explore'
        mode2 = 'explore'
        action_1 = self.agent_1.act(observation_n,mode1)
        action_2 = self.agent_2.act(observation_n,mode2)
        action_n = action_1 + action_2
        
        observation_n, reward_n, done_n, info = self.env.step(action_n)
        
        
        global_state = observation_n
        #self.total_rewards[0] += sum(reward_n[0:2])
        #self.total_rewards[1] += sum(reward_n[3:5])
        return (old_observation_n, action_n, old_global_state, reward_n,
                     observation_n, global_state, done_n)
    
    #populate the buffer with a full episode
    def roll_in_episode(self): # -> with this operation we could do a pre-filling procedure
        #roll in phase
        id = 0
        done = False
        while (id<self.episode_limit and not done):

            (o, a, s, r, o_next, s_next, d)= self.make_step( self.observation_n)
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


    
if __name__ == '__main__':
    agent = Agents()
    agent.roll_in_episode()
    sample=agent.buffer.sample(2)
    print(sample)
    print('daje brah')
