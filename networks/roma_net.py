import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import kl_divergence
import torch.distributions as D
import math

class RomaAgent(nn.Module):
    def __init__(self,input_shape,
                      n_agents,
                      n_actions,
                      latent_dim,
                      rnn_hidden_dim,
                      batch_size,
                      fc_hidden_size=12
                      ):
        super(RomaAgent,self).__init__()
            
        self.input_shape = input_shape
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.batch_size = batch_size #bs
        self.fc_hidden_size=fc_hidden_size
        self.activation=nn.LeakyReLU()
        #self.role_encoder
        self.embed_net=nn.Sequential(nn.Linear(input_shape,self.fc_hidden_size),
                                     #nn.BatchNorm1d(self.fc_hidden_size),
                                     self.activation,
                                     nn.Linear(self.fc_hidden_size,self.latent_dim*2) #mu + var
                                        )
        #self.role_decoder
        self.inference_net = nn.Sequential(nn.Linear(self.rnn_hidden_dim+input_shape,self.fc_hidden_size),
                                            self.activation,
                                            nn.Linear(self.fc_hidden_size,self.latent_dim*2)                              
                                            )

        
        self.latent = torch.rand(self.n_agents, self.latent_dim * 2)  # (n,mu+var)
        self.latent_infer = torch.rand(self.n_agents, self.latent_dim * 2)  # (n,mu+var)

        self.latent_net = nn.Sequential(nn.Linear(self.latent_dim, self.fc_hidden_size),
                                        #nn.BatchNorm1d(self.fc_hidden_size),
                                        self.activation)
                            
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        self.fc2_w_nn = nn.Linear(self.fc_hidden_size, self.rnn_hidden_dim * self.n_actions)
        self.fc2_b_nn = nn.Linear(self.fc_hidden_size, self.n_actions)

        
        # Dissimilarity Net
        self.dis_net = nn.Sequential(nn.Linear(self.latent_dim * 2, self.fc_hidden_size ),
                                     #nn.BatchNorm1d(self.fc_hidden_size ),
                                     self.activation,
                                     nn.Linear(self.fc_hidden_size , 1))

        self.mi= torch.rand(self.n_agents*self.n_agents)
        self.dissimilarity = torch.rand(self.n_agents*self.n_agents)
