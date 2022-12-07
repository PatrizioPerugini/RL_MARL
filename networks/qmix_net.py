import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Qmix_Net(nn.Module):
    def __init__(self, state_shape, n_agents, hidden_dim ):
        super(Qmix_Net, self).__init__()

        self.state_shape = state_shape
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        #assuming we work with derk we have state_shape [6,64]-> hence stat_ dim ->6*64
        self.state_dim = int(np.prod(self.state_shape))              
        
        #architecture -hyper networks
        self.hyper_w1 = nn.Linear(self.state_dim, self.hidden_dim * self.n_agents) 
        self.hyper_w2 = nn.Linear(self.state_dim, self.hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, 1))
        
    
    def forward(self, q_values, states):  
        # q_values.shape = (batch_size (bs), trajectory_len (t)， n_agents)
        # states.shape = (batch_size (bs), trajectory_len (t), state_dim)

        batch_size = q_values.size(0)       
        traj_len = q_values.size(1) #the trajectory len in the output h of the RNN agent

        q_values = q_values.view(-1, 1, self.n_agents)      # (bs*t , 1, n_agents)
        states = states.reshape(-1, self.state_dim)            # (bs*t , state_dim)


        # Layer 1
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(-1, self.n_agents, self.hidden_dim)        # (bs*t , n_agents, hidden_dim)

        b1 = self.hyper_b1(states)
        b1 = b1.view(-1, 1, self.hidden_dim)         # (bs*t , 1, hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)        # (bs*t , 1, hidden_dim)


        # Layer 2

        w2 = torch.abs(self.hyper_w2(states))  # (bs*t , hidden_dim)
        w2 = w2.view(-1, self.hidden_dim, 1)  # (bs*t , hidden_dim, 1)

        b2 = self.hyper_b2(states)  # (bs*t , 1)
        b2 = b2.view(-1, 1, 1)  # (bs*t , 1， 1)


        # Output layer
        q_total = torch.bmm(hidden, w2) + b2  # (bs*t , 1, 1)
        q_total = q_total.view(batch_size, -1, 1)  # (bs, t, 1)

        return q_total























