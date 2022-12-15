import torch
from gym_derk.envs import DerkEnv
from utils.prova_buffer import ReplayBuffer
from networks.rnn_net import RNNAgent
from networks.qmix_net import Qmix_Net
import random
from copy import copy
import numpy as np
import torch.nn as nn


#away team.. lazio
class Agent_Qmix():

    def __init__(self,custom_env,team):#,replay_buffer):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #self.device = torch.device("cpu")
        
        self.env = custom_env.env
        self.action_space = custom_env.action_space
        self.team = team
        self.n_agents = int(self.env.n_agents / 2)    #teams have the same number of components
        self.team_members_id = self.team_split([i for i in range(6)])
        self.n_actions = self.action_space.count
        self.state_shape = self.env.observation_space.shape[0]
        self.action_dict = self.action_space.actions

        self.RNN_input_shape = self.state_shape + self.action_space.action_len
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
        self.epsilon=0.9
        self.epsilon_decay=0.999
        self.epsilon_treshold=0.07
        self.cnt_update=0
        self.update_freq=10
        
        self.batch_size = 32
        
        #optimize multiple net
        params = list(self.qmix.parameters()) + list(self.rnn_1.parameters()) \
                        +list(self.rnn_1.parameters())+ list(self.rnn_1.parameters())

        self.optimizer = torch.optim.Adam(params,
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
            actions = self.action_space.sample()[0]
        return actions
    
    def reset_hidden_states(self):
        for i in range(len(self.rnn_agents)):
            self.hidden_states[i]= self.rnn_agents[i].init_hidden().to(self.device)
        
        

    def update(self,buffer,episode_limit=150):
        print("training team number",self.team)
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
            last_action_id = np.zeros((3))     # 4 is the index of (0,0,0,0,0)
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
                
                q_f = torch.cat((qvals[0],qvals[1],qvals[2]),dim=0).to(self.device)
                q_f = torch.gather(q_f,1,torch.LongTensor(actions_id.reshape(-1,1)).to(self.device)).squeeze(-1)

                stack_batch_qvals[i,t,:]=q_f

                last_action_id = actions_id
           
            stack_batch_next_qvals[i,:-1]=stack_batch_qvals[i,1:]
            #maybe we should reconsider this... and just set them to zero 
            if traj_len < episode_limit:
                stack_batch_state[i,traj_len:,:] = torch.from_numpy(global_state)
                stack_batch_next_state[i,traj_len:,:]=torch.from_numpy(next_gs)
                stack_batch_terminated[i,traj_len:] = 1

            for j in range(len(self.rnn_agents)):
                a = np.array(self.action_dict[int(actions_id[j])])
                input_rnn = np.hstack((obs_next[j].astype('float32'),a.astype('float32')))
                input_rnn=torch.from_numpy(input_rnn).unsqueeze(0).to(self.device)
                next_qvals[j], self.hidden_states[j] = self.rnn_agents[j].forward(input_rnn,self.hidden_states[j])
            
            next_q_f = torch.cat((next_qvals[0],next_qvals[1],next_qvals[2]),dim=0)
            next_q_f_max = torch.max(next_q_f,dim =-1)[0]#.reshape(-1,1)
            #next_q_f = torch.gather(next_q_f,1,torch.LongTensor(actions_id.reshape(-1,1))).squeeze(-1)

            stack_batch_next_qvals[i,-1]=next_q_f_max

        #we have all the q vals and the states to feed to qmix
        #in the variables stack_batch_qvals and stack_batch_state

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
        
        if self.epsilon>self.epsilon_treshold:
            self.epsilon*=self.epsilon_decay
        if self.cnt_update%self.update_freq==0:
            self.update_target_q_net()
        self.cnt_update+=1
    
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
   
    
    def save(self):
        torch.save(self.qmix.state_dict(),  "model_qmix" +"_"+ str(self.team)+".pt")
        torch.save(self.rnn_1.state_dict(), "model_rnn1" +"_"+ str(self.team)+".pt")
        torch.save(self.rnn_2.state_dict(), "model_rnn2" +"_"+ str(self.team)+".pt")
        torch.save(self.rnn_3.state_dict(), "model_rnn3" +"_"+ str(self.team)+".pt")
    
    def load(self):
        self.qmix.load_state_dict(torch.load("model_qmix.pt", map_location=self.device))
        self.rnn_1.load_state_dict(torch.load("model_rnn1.pt", map_location=self.device))
        self.rnn_2.load_state_dict(torch.load("model_rnn2.pt", map_location=self.device))
        self.rnn_3.load_state_dict(torch.load("model_rnn3.pt", map_location=self.device))
    
    def update_target_q_net(self):
        self.target_qmix.load_state_dict(self.qmix.state_dict())