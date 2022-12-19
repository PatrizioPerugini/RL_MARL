import torch
from gym_derk.envs import DerkEnv
from utils.reply_buffer import ReplayBuffer
from Roma_brain import Agent_ROMA
from Qmix_brain import Agent_RNN
import random
import numpy as np
from matplotlib import pyplot 
import time
import datetime

import warnings

warnings.filterwarnings('ignore')

class Agents():

    def __init__(self,custom_env,vs):  
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = custom_env.env
        #agent 1 is trained by roma
        
              
        if vs == 'RvsR': 
            self.match = ['roma','roma']
        elif vs == 'RvsQ':
            self.match = ['roma','qmix']
        elif vs == 'QvsQ':
            self.match = ['qmix','qmix']
        elif vs == 'RvsG':
            self.match = ['roma','ghost']
        elif vs == 'QvsG':
            self.match = ['qmix','ghost']
        else:
            raise Exception('Models not valid!')

        print(self.match)
            

        self.action_space = custom_env.action_space
        self.action_dict = self.action_space.actions
        self.action_shape = 5
        self.n_agents = self.env.n_agents
        self.obs_shape = self.env.observation_space.shape[0]
        self.state_shape = self.env.observation_space.shape[0]
        self.last_action = np.zeros((6,5))

        # global vars
        self.observation_n = self.env.reset()
        self.training_epochs=100

        #buffer 
        self.episode_limit = 151
        self.buffer_size = 500
        self.batch_size = 16
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

        #agent instantiation
        self.ghost = False
        if self.match[0] == 'roma':
            self.agent_1 = Agent_ROMA(custom_env,1, self.batch_size,vs)
        else:
            self.agent_1 = Agent_RNN(custom_env,1, self.batch_size,vs)

        if self.match[1] == 'roma':
            self.agent_2 = Agent_ROMA(custom_env,2, self.batch_size,vs)
        elif self.match[1] == 'qmix':
            self.agent_2 = Agent_RNN(custom_env,2, self.batch_size,vs)
        elif self.match[1] == 'ghost':
            self.agent_2 = Ghost()
            self.ghost = True


        # simulation
        self.total_rewards = [0,0]
        self.update_loss_1 = []
        self.update_loss_2 = []



    def make_step(self, observation_n, last_action_n, exploit = [True,True] ):

        old_observation_n = observation_n 
        old_global_state = old_observation_n
        
        input_rnn = np.hstack((observation_n.astype('float32'),last_action_n.astype('float32')))

        action_1_id = self.agent_1.act(input_rnn,exploit[0])
        action_2_id = self.agent_2.act(input_rnn,exploit[1])
        action_n_id = action_1_id + action_2_id
        action_to_do = [ self.action_dict[idx] for idx in action_n_id]
        observation_n, reward_n, done_n, info = self.env.step(action_to_do)
        
        global_state = observation_n
        #self.total_rewards[0] += sum(reward_n[0:2])
        #self.total_rewards[1] += sum(reward_n[3:5])
        return (old_observation_n, action_n_id, old_global_state, reward_n,
                     observation_n, global_state, done_n)
    

    def e_choice(self,agent):
        
        p = random.random()
        if p<agent.epsilon:
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
        self.agent_1.reset_hidden_states(1)
        
        #self.agent_2.reset_hidden_states(1)
        if not self.ghost:
            self.agent_2.reset_hidden_states()
        while (id<self.episode_limit and not done):
            #there is no point in exploiting an untrained agent
            #you would only unlearn
            if training_ag=='lazio':
                #e_exploit = [True,self.e_choice(self.agent_2)]
                e_exploit = [self.e_choice(self.agent_1),self.e_choice(self.agent_2)]
            else:
                #e_exploit = [self.e_choice(self.agent_1),True]
                e_exploit = [self.e_choice(self.agent_1),self.e_choice(self.agent_2)]

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

        
        
        #print('Batch saved in the buffer.')
    
    #max_steps must be greater then bs
    def train(self,max_steps=5,episodes=3):
        start = time.time()
        end_old = start
        print("\nSTART training")

        print('\n     Roma as:',self.match[0])
        print('          Vs')
        print('     Lazio as:',self.match[1])

        print('\nFirst roll in some episodes ...')
        for r_i in range(int(self.batch_size/2)):
            self.roll_in_episode("roma")
        for r_i in range(int(self.batch_size/2)):
            self.roll_in_episode("lazio")

        
        for epochs in range(self.training_epochs):
            print("\nEPOCH: ",epochs)
            print('---------------------------------------------')
            for ep in range(episodes):
                print("T1 Episode:",epochs,'.',ep)
                for r_i in range(max_steps):
                    #print("\r - Rolling in episode {:d}...".format(r_i+1),end="")
                    self.roll_in_episode("roma")
                
                loss = self.agent_1.update(self.buffer,self.episode_limit)
                self.update_loss_1.append(loss)
                print(" - Loss: ",loss )
                print(' - Epsilon:',self.agent_1.epsilon)
                print(' - Buffer capacity:', self.buffer.get_capacity())

            if not self.ghost:
                for ep in range(episodes):
                    print("T2 Episode:",epochs,'.',ep)
                    for f_i in range(max_steps):
                        #print("\r - Rolling in episode {:d} ...".format(f_i+1),end="")
                        self.roll_in_episode("lazio")
                    
                    loss = self.agent_2.update(self.buffer,self.episode_limit)
                    self.update_loss_2.append(loss)
                    print(" - Loss: ",loss )
                    print(' - Epsilon:',self.agent_2.epsilon)
                    print(' - Buffer capacity:', self.buffer.get_capacity())

            end = time.time()
            duration = end-end_old

            print('Epoch duration:',str(datetime.timedelta(seconds=duration)))
            end_old = end
            print('***************** SMACK DOWN *****************')
            self.evaluation()
            print('**********************************************')

            continue

        print("END training")
        duration = time.time() - start
        print('Total training duration:',str(datetime.timedelta(seconds=duration)))
        pyplot.plot(self.update_loss_1)
        pyplot.plot(self.update_loss_2)



    

    def evaluation(self):
        observation_n = np.array(self.observation_n)
        last_action = np.zeros((6,5))
        self.agent_1.reset_hidden_states(1)
        if not self.ghost:
            self.agent_2.reset_hidden_states(1)

        done = False
        rewards_1 =0
        rewards_2 =0

        while not done:
            input_rnn = np.hstack((observation_n.astype('float32'),last_action.astype('float32')))
            
            action_1_id = self.agent_1.act(input_rnn,True)
            action_2_id = self.agent_2.act(input_rnn,True)
            action_n_id = action_1_id + action_2_id
           
            action_to_do = [ self.action_dict[idx] for idx in action_n_id]
            observation_n, reward_n, done_n, _ = self.env.step(action_to_do)
            #observation_n, reward_n, done_n, _ = self.env.step(action_to_do)
            self.last_action =action_to_do
            rewards_1 += reward_n[0:3]
            rewards_2 += reward_n[3:6]
            done = done_n[0]
        self.print_stats()
        self.observation_n=self.env.reset()
        return 1

    def print_stats(self):
        for i in [0,1]:
            print('TEAM',i+1)
            print(' - Reward:',self.env.team_stats[i,0])
            print(' - Hitpoints:',self.env.team_stats[i,2])
            print(' - AliveTime:',self.env.team_stats[i,3])
            print(' - CumulativeHitpoints:',self.env.team_stats[i,4])


class Ghost():

    def __init__(self):
        self.epsilon=0

    def act(self,input_rnn,exploit):
        return [0,0,0]
    def update(self,buffer,episode_limit):
        return 0
    def load(self):
        pass



