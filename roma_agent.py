import torch
from gym_derk.envs import DerkEnv
from utils.prova_buffer import ReplayBuffer
from networks.roma_net import RomaAgent
from networks.qmix_net import Qmix_Net

import random
from copy import deepcopy
import numpy as np
import torch.nn as nn



class Agent_ROMA():

    def __init__(self,custom_env,team):#,replay_buffer):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.device = torch.device("cpu")
        
        self.env = custom_env.env
        self.action_space = custom_env.action_space
        self.team = team
        self.n_agents = int(self.env.n_agents / 2)    #teams have the same number of components
        self.team_members_id = self.team_split([i for i in range(6)])
        self.n_actions = self.action_space.count
        self.state_shape = self.env.observation_space.shape[0]
        self.action_dict = self.action_space.actions
        self.batch_size = 2

        self.input_shape = self.state_shape + self.action_space.action_len
        self.rnn_hidden_dim = 32
        self.hidden_state = None
        self.target_hidden_state = None
        self.ROMA_agent=RomaAgent(self.input_shape,
                      self.n_agents,
                      self.n_actions,
                      latent_dim=12,
                      rnn_hidden_dim=self.rnn_hidden_dim,
                      batch_size=self.batch_size)
        
        self.target_ROMA_agent = deepcopy(self.ROMA_agent)
                      

        self.Qmix_hidden_dim = 32
        self.qmix = Qmix_Net((self.n_agents,self.state_shape), self.n_agents, self.Qmix_hidden_dim).to(self.device)
        self.target_qmix = Qmix_Net((self.n_agents,self.state_shape), self.n_agents, self.Qmix_hidden_dim).to(self.device)
  
        
        # Training stuff
        #self.tot_episodes = 500
        #self.max_steps_per_episode = 1000
        self.gamma=0.99
        self.learning_rate=0.00025
        self.epsilon=0.3
   
        self.reset_hidden_states(self.batch_size)
                #optimize multiple net
        
        params = list(self.qmix.parameters()) + list(self.ROMA_agent.parameters())
        
        self.optimizer = torch.optim.Adam(params,
                                          lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def greedy_action(self,input):
        input = self.team_split(input)
        input = torch.from_numpy(input).unsqueeze(0)
        input= input.to(self.device)
        indices,self.hidden_state = self.ROMA_agent.greedy_action_id(input,self.hidden_state)
        print("succesfully achieved")
        return indices


    def team_split(self,obj):
        if self.team ==1:
            return obj[0:3]
        else:
            return obj[3:6]
    
    def act(self, input, exploit):  
        input = self.team_split(input)
        #actions = self.greedy_action(input)
        actions = self.action_space.sample()[0]
        if not exploit:
            actions = self.action_space.sample()[0]
        return actions
    
    def reset_hidden_states(self,batch_size = 1):
        self.hidden_state = self.ROMA_agent.fc1.weight.new(batch_size,self.n_agents, self.rnn_hidden_dim).zero_()
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

        return torch.Tensor(np.array(inputs)) , torch.LongTensor(np.array(a_id))
        

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

        return torch.Tensor(np.array(inputs))


    def update(self,buffer,episode_limit=150):
        print("training team number",self.team)

        self.reset_hidden_states(self.batch_size)
        #self.load()
        #batch = buffer.sample(self.batch_size)
        stack_batch_qvals=torch.zeros((self.batch_size,episode_limit,self.n_agents)).to(self.device)#*-9999#(1,episode_limit,agents))
        
        #(bs,traj,6,64)
        stack_batch_next_qvals=torch.zeros((self.batch_size,episode_limit,self.n_agents)).to(self.device)
        batch = buffer.sample(self.batch_size)
        
        traj_len_max = len(batch['o'][0])

        stack_batch_state,stack_batch_next_state, stack_batch_rewards, stack_batch_terminated =\
            self.build_stack(batch,episode_limit)
        stack_batch_qvals = torch.zeros((self.batch_size,traj_len_max,self.n_agents))

        loss = 0     #regularization
        d_loss = 0        #dissimilary
        c_loss = 0       #crossentropy

        for t in range(traj_len_max): #trajectory len
           
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
            q_f = torch.gather(q_f.reshape(-1,16),1,actions_id.reshape(-1,1)).to(self.device).squeeze(-1)
            q_f = q_f.reshape(-1,self.n_agents)

            stack_batch_qvals[:,t,:] = q_f
        

        q_tot = self.qmix.forward(stack_batch_qvals,stack_batch_state).squeeze(-1)
        
        loss /= traj_len_max
        d_loss /= traj_len_max
        c_loss /= traj_len_max

        self.reset_hidden_states(batch_size = self.batch_size)

        for t in range(traj_len_max): #trajectory len
           
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
        #td_error = (q_tot.detach()- target_qtot)
        loss += TD_loss 
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
        torch.save(self.qmix.state_dict(),  "model_qmix" +"_"+ str(self.team)+".pt")
        torch.save(self.ROMA_agent.state_dict(),  "model_ROMA_agent" +"_"+ str(self.team)+".pt")

    
    def load(self):
        self.qmix.load_state_dict(torch.load("model_qmix" +"_"+ str(self.team)+".pt", map_location=self.device))
        self.ROMA_agent.load_state_dict(torch.load("model_ROMA_agent" +"_"+ str(self.team)+".pt", map_location=self.device))
    
    def update_target_q_net(self):
        self.target_qmix.load_state_dict(self.qmix.state_dict())
        self.target_ROMA_agent.load_state_dict(self.ROMA_agent.state_dict())




class Agents():

    def __init__(self,custom_env):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = custom_env.env
        self.agent_1 = Agent_ROMA(custom_env,team = 1)
        self.agent_2 = Agent_ROMA(custom_env,team = 2)
        self.action_space = custom_env.action_space
        self.action_dict = self.action_space.actions
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
        self.agent_1.reset_hidden_states(1)
        self.agent_2.reset_hidden_states(1)


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
    def train(self,max_steps=3,episodes=3):
        #while something do
        for ep in range(episodes):
            for r_i in range(max_steps):
                self.roll_in_episode(0)
                #self.observation_n=self.env.reset()
            self.agent_1.update(self.buffer,self.episode_limit)
        for ep in range(episodes):
            for _ in range(max_steps):
                self.roll_in_episode(1)
            self.agent_2.update(self.buffer,self.episode_limit)
        #loop


