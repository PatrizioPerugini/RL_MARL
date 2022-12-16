import torch
from gym_derk.envs import DerkEnv
from utils.reply_buffer import ReplayBuffer
from networks.roma_net import RomaAgent
from networks.qmix_net import Qmix_Net

import random
from copy import deepcopy
import numpy as np
import torch.nn as nn

import warnings

warnings.filterwarnings('ignore')

#home team ... r
class Agent_ROMA():

    def __init__(self,custom_env,team,batch_size):#,replay_buffer):
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
        self.batch_size = batch_size

        self.input_shape = self.state_shape + self.action_space.action_len
        self.rnn_hidden_dim = 32
        self.hidden_state = None
        self.target_hidden_state = None
        self.ROMA_agent=RomaAgent(self.input_shape,
                      self.n_agents,
                      self.n_actions,
                      latent_dim=12,
                      rnn_hidden_dim=self.rnn_hidden_dim,
                      batch_size=self.batch_size).to(self.device)
        
        self.target_ROMA_agent = deepcopy(self.ROMA_agent)
                      

        self.Qmix_hidden_dim = 32
        self.qmix = Qmix_Net((self.n_agents,self.state_shape), self.n_agents, self.Qmix_hidden_dim).to(self.device)
        self.target_qmix = Qmix_Net((self.n_agents,self.state_shape), self.n_agents, self.Qmix_hidden_dim).to(self.device)
  
        
        # Training stuff
        #self.tot_episodes = 500
        #self.max_steps_per_episode = 1000
        self.cnt_update=0
        self.update_freq=5
        self.gamma=0.99
        self.learning_rate=0.00025
        self.epsilon=0.85
        self.epsilon_decay=0.999
        self.epsilon_treshold=0.07
   
        self.reset_hidden_states(self.batch_size)
                #optimize multiple net
        
        params = list(self.qmix.parameters()) + list(self.ROMA_agent.parameters())
        
        self.optimizer = torch.optim.Adam(params,
                                          lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def greedy_action(self,input):
        input = torch.from_numpy(input).unsqueeze(0)
        input= input.to(self.device)
        indices,self.hidden_state = self.ROMA_agent.greedy_action_id(input,self.hidden_state)
        return indices.squeeze(0).tolist()


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
    
    def reset_hidden_states(self,batch_size = 1):
        self.hidden_state = self.ROMA_agent.fc1.weight.new(batch_size,self.n_agents, self.rnn_hidden_dim).zero_().to(self.device)
        self.target_hidden_state= self.target_ROMA_agent.fc1.weight.new(batch_size,self.n_agents, self.rnn_hidden_dim).zero_()
        



    def build_inputs(self, batch, t):
        inputs = []
        obs = batch["o"][:, t] 
        a_id = batch["a"][:,t]
        if t == 0:
            l_action_ids = np.zeros((self.batch_size,6))
        else:
            l_action_ids = batch["a"][:, t - 1]

        inputs=[]
        for j in range(self.batch_size):
            l_action = [self.action_dict[x] for x in l_action_ids[j]]
            
            inputs.append(np.hstack((obs[j].astype('float32'),np.array(l_action).astype('float32'))))

        return torch.Tensor(np.array(inputs)).to(self.device) \
                , torch.LongTensor(np.array(a_id)).to(self.device)
        

    def build_stack(self,batch,episode_limit):
        stack_batch_state = torch.zeros(self.batch_size,episode_limit,self.n_agents*2,self.state_shape).to(self.device)
        stack_batch_next_state = torch.zeros(self.batch_size,episode_limit,self.n_agents*2,self.state_shape).to(self.device)
        stack_batch_rewards = torch.zeros(self.batch_size,episode_limit,self.n_agents).to(self.device)
        stack_batch_terminated = torch.zeros(self.batch_size,episode_limit).to(self.device)
        for i in range(self.batch_size):
            traj_len = int(batch['episode_len'][i])
            for t in range(traj_len):
                rewards = self.team_split(batch['r'][i][t]) #(3,)
                global_state=batch['s'][i][t]
                next_gs = batch['s_next'][i][t]
                terminated = batch['terminated'][i][t]

                stack_batch_state[i,t] = torch.from_numpy(global_state)
                stack_batch_next_state[i,t] = torch.from_numpy(next_gs)
                stack_batch_rewards[i,t] = torch.from_numpy(rewards)
                stack_batch_terminated[i,t] =torch.from_numpy(terminated)
            
        return stack_batch_state,stack_batch_next_state,stack_batch_rewards,stack_batch_terminated


    def build_next_inputs(self, batch, t):
        inputs = []
        next_obs = batch["o_next"][:, t] 
        
        l_action_ids = batch["a"][:, t]

        inputs=[]
        for j in range(self.batch_size):
            l_action = [self.action_dict[x] for x in l_action_ids[j]]
            
            inputs.append(np.hstack((next_obs[j].astype('float32'),np.array(l_action).astype('float32'))))

        return torch.Tensor(np.array(inputs)).to(self.device)


    def update(self,buffer,episode_limit=150):
        self.reset_hidden_states(self.batch_size)
        self.load('models_RVsR')
        #batch = buffer.sample(self.batch_size)
        stack_batch_qvals=torch.zeros((self.batch_size,episode_limit,self.n_agents)).to(self.device)#*-9999#(1,episode_limit,agents))
        
        #(bs,traj,6,64)
        stack_batch_next_qvals=torch.zeros((self.batch_size,episode_limit,self.n_agents)).to(self.device)
        batch = buffer.sample(self.batch_size)

        stack_batch_state,stack_batch_next_state, stack_batch_rewards, stack_batch_terminated =\
            self.build_stack(batch,episode_limit)
        

        loss = 0     #regularization
        d_loss = 0        #dissimilary
        c_loss = 0       #crossentropy

        for t in range(episode_limit): #trajectory len
           
            stack_inputs, actions_id = self.build_inputs(batch, t)

            #stack inputs shape after split -> (bs,n_agents,obs+act)
            if self.team ==1:
                stack_inputs = stack_inputs[:,0:3,:]
                actions_id =actions_id[:,0:3]
            else: 
                stack_inputs = stack_inputs[:,3:6,:]
                actions_id =actions_id[:,3:6]

            #shape of q is -> (bs,n_agents_n_actions)
            q_f,self.hidden_state,loss_,d_loss_,c_loss_ = self.ROMA_agent.forward(stack_inputs,self.hidden_state,train_mode=True)
            loss += loss_
            d_loss += d_loss_
            c_loss += c_loss_
            q_f = torch.gather(q_f.reshape(-1,self.n_actions),1,actions_id.reshape(-1,1)).to(self.device).squeeze(-1)
            q_f = q_f.reshape(-1,self.n_agents)

            stack_batch_qvals[:,t,:] = q_f
        

        q_tot = self.qmix.forward(stack_batch_qvals,stack_batch_state).squeeze(-1)
        
        
        loss /= episode_limit
        d_loss /= episode_limit
        c_loss /= episode_limit

        self.reset_hidden_states(batch_size = self.batch_size)

        for t in range(episode_limit): #trajectory len
           
            stack_next_inputs = self.build_next_inputs(batch, t)

            #stack inputs shape after split -> (bs,n_agents,obs+act)
            if self.team ==1:
                stack_next_inputs = stack_next_inputs[:,0:3,:]
            else: 
                stack_next_inputs = stack_next_inputs[:,3:6,:]

            #shape of q is -> (bs,n_agents_n_actions)
            next_q_f,self.hidden_state,_,_,_ = self.target_ROMA_agent.forward(stack_next_inputs,self.hidden_state,train_mode=False)
            next_q_f=next_q_f.reshape(-1,self.n_agents,self.n_actions)

            next_q_f_max = torch.max(next_q_f,dim =-1)[0]
            stack_batch_next_qvals[:,t,:] = next_q_f_max

        #print(q_tot.shape) -> (bs,t,1)
        next_qtot_max = self.target_qmix(stack_batch_next_qvals,stack_batch_next_state)

        rews = stack_batch_rewards.sum(-1) 
        target_qtot = rews + (1-stack_batch_terminated)*next_qtot_max.squeeze(-1)*self.gamma
        
        TD_loss = self.loss_function(q_tot.detach(), target_qtot)
       
        loss += TD_loss 
        
        
        loss.backward()
        self.optimizer.step()
        #self.update_loss.append(loss.item())
        self.save('models_RVsR')
        if self.cnt_update%self.update_freq==0:

            self.update_target_q_net()
        self.cnt_update+=1
        if self.epsilon>self.epsilon_treshold:
            self.epsilon*=self.epsilon_decay

        return loss.item()

    
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
   
    
    def save(self,where):
        torch.save(self.qmix.state_dict(),  where+"/model_qmix" +"_"+ str(self.team)+".pt")
        torch.save(self.ROMA_agent.state_dict(),  where+"/model_ROMA_agent" +"_"+ str(self.team)+".pt")

    
    def load(self,where):
        self.qmix.load_state_dict(torch.load(where+"/model_qmix" +"_"+ str(self.team)+".pt", map_location=self.device))
        self.ROMA_agent.load_state_dict(torch.load(where+"/model_ROMA_agent" +"_"+ str(self.team)+".pt", map_location=self.device))
    
    def update_target_q_net(self):
        self.target_qmix.load_state_dict(self.qmix.state_dict())
        self.target_ROMA_agent.load_state_dict(self.ROMA_agent.state_dict())