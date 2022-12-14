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
                      fc_hidden_size=12,
                      var_floor=0.1,
                      h_loss_weight=0.5,
                      kl_loss_weight=0.5,
                      dis_loss_weight=0.01
                      ):
        super(RomaAgent,self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.latent_dim = latent_dim #dimension of the latent space
        self.rnn_hidden_dim = rnn_hidden_dim
        self.batch_size = batch_size #bs
        #self.batch_size = batch_size
        self.fc_hidden_size=fc_hidden_size #hidden size of each mlp
        self.embed_fc_input_size = input_shape
        self.var_floor = var_floor
        self.activation=nn.LeakyReLU()

        self.h_loss_weight = h_loss_weight
        self.kl_loss_weight=kl_loss_weight
        self.dis_loss_weight=dis_loss_weight
        #----------------------------------------------------------------------------------------
        #ROLE_ENCODER
        #OK, EMBED NET PROJECTS THE INPUT INTO A LATENT SPACE OF MU AND VAR                                
                                                #IT'S THE INPUT SHAPE
        self.embed_net=nn.Sequential(nn.Linear(self.embed_fc_input_size,self.fc_hidden_size),
                                     #nn.BatchNorm1d(self.fc_hidden_size),
                                     self.activation,
                                     nn.Linear(self.fc_hidden_size,self.latent_dim*2) #mu + var
                                        )
        #----------------------------------------------------------------------------------------
        #TRAJECTORY ENCODER
        self.inference_net = nn.Sequential(nn.Linear(self.rnn_hidden_dim+input_shape,self.fc_hidden_size),
                                            self.activation,
                                            nn.Linear(self.fc_hidden_size,self.latent_dim*2)                              
                                            )

        
        self.latent = torch.rand(self.n_agents, self.latent_dim * 2)  # (n,mu+var)
        self.latent_infer = torch.rand(self.n_agents, self.latent_dim * 2)  # (n,mu+var)
        
        #----------------------------------------------------------------------------------------
        #ROLE_DECODER 
        #OK, IT TAKES THE ROLES SAMPLED FROM THE LATENT SPACE AS INPUT 
        # AND CONDITION WITH IT'S OUTPUT WILL CONDITION THE RNN AGENT
        self.latent_net = nn.Sequential(nn.Linear(self.latent_dim, self.fc_hidden_size),
                                        #nn.BatchNorm1d(self.fc_hidden_size),
                                        self.activation)

        
        #----------------------------------------------------------------------------------------
        #RNN AGENT -> fc,gru,fc(which is given by an hypernet)               
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        
        #self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.GRU_num_layers = 1
        self.rnn = nn.GRU(input_size=self.rnn_hidden_dim,
             hidden_size=self.rnn_hidden_dim,
             num_layers=self.GRU_num_layers,
             batch_first=True,
             bidirectional=False)

        #output of latent_net is the input of self.fc2_w_nn
        #this is how we are able to codition the on the actual role
        self.fc2_w_nn = nn.Linear(self.fc_hidden_size, self.rnn_hidden_dim * self.n_actions)
        self.fc2_b_nn = nn.Linear(self.fc_hidden_size, self.n_actions)

        
        #----------------------------------------------------------------------------------------
        #THIS IS JUST ANOTHER NETWORK TO COMPUTE THE DISSIMILARITY LOSS...NOTHING SPECTACULAR
        # Dissimilarity Net                    #latent_dim=rnn_hidden_dim=T
        #V
        self.dis_net = nn.Sequential(nn.Linear(self.latent_dim * 2, self.fc_hidden_size ),
                                     #nn.BatchNorm1d(self.fc_hidden_size ),
                                     self.activation,
                                     nn.Linear(self.fc_hidden_size , 1)) # T x T -> R

        self.mi= torch.rand(self.n_agents*self.n_agents)
        self.dissimilarity = torch.rand(self.n_agents*self.n_agents)

        #self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_sigmoid

    def init_latent(self,batch_size): #->indicators, latent,latent_infer
        self.batch_size=batch_size
        
        self.trajectory=[]
                          #same as :
        var_mean=self.latent[:self.n_agents, self.latent_dim:].detach().mean()
        mi = self.mi
        di = self.dissimilarity
        indicator=[var_mean,mi.max(),mi.min(),mi.mean(),mi.std(),
                    di.max(),di.min(),di.mean(),di.std()]
        return indicator, self.latent[:self.n_agents, :].detach(), self.latent_infer[:self.n_agents, :].detach()

    def forward(self,inputs,hidden_state,t=0,train_mode=True,t_glob=0):
        bs = inputs.shape[0]

        inputs = inputs.reshape(-1, self.input_shape) # bs*num_ag,obs+ac
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)  #bs*num_ag, rnn_hd
                                    #input_shape..se pia tutto
        
        #nothing
        embed_fc_input = inputs[:, - self.embed_fc_input_size:]  # own features(unit_type_bits+shield_bits_ally)+id
        #encode everithing-> mu ,var
       
        self.latent = self.embed_net(embed_fc_input) #(n_agents,latent_dim*2)
        
        self.latent[:, -self.latent_dim:] = torch.clamp(torch.exp(self.latent[:, -self.latent_dim:]), min=self.var_floor)  # var
        #shape is (bs*n_ag, latent_dim*2)       
        latent_embed = self.latent.reshape(bs * self.n_agents, self.latent_dim * 2)
        
        #shape is (bs*n_ag, latent_dim)
                                                    #media                              #var
        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2))
        
        #shape is (bs*n_ag, latent_dim
        latent = gaussian_embed.rsample() #we can back propagate with rsample

        c_dis_loss = torch.tensor(0.0).to(self.device)
        ce_loss = torch.tensor(0.0).to(self.device)
        loss = torch.tensor(0.0).to(self.device)

        if train_mode:
            #shape is (bs*n_ag, latent_dim*2)


            self.latent_infer = self.inference_net(torch.cat([h_in.detach(), inputs], dim=1))
            self.latent_infer[:, -self.latent_dim:] = torch.clamp(torch.exp(self.latent_infer[:, -self.latent_dim:]),min=self.var_floor)
            
            gaussian_infer = D.Normal(self.latent_infer[:, :self.latent_dim], (self.latent_infer[:, self.latent_dim:]) ** (1 / 2))
            
            #shape is (bs*n_ag, latent_dim)
            latent_infer = gaussian_infer.rsample() #just to have a view

            h_loss=gaussian_embed.entropy().sum(dim=-1).mean()
            kl_loss=kl_divergence(gaussian_embed, gaussian_infer).sum(dim=-1).mean()
            # CE = H + KL
            loss = h_loss* self.h_loss_weight + kl_loss* self.kl_loss_weight
                       
            loss = torch.clamp(loss, max=2e3)
            # loss = loss / (self.bs * self.n_agents)
            ce_loss = torch.log(1 + torch.exp(loss))


        #-----------------------loss 1 done -----------------------
            dis_loss = 0
            dissimilarity_cat = None
            mi_cat = None
            
            #shape is (bs,n_agents,latent_dim)
            latent_dis = latent.clone().view(bs, self.n_agents, -1)
            latent_move = latent.clone().view(bs, self.n_agents, -1)

            for agent_i in range(self.n_agents):
                    latent_move = torch.cat([latent_move[:, -1, :].unsqueeze(1), 
                                            latent_move[:, :-1, :]], dim=1)
                    
                    #shape is (bs,n_agents,2*latent_dim)
                    latent_dis_pair = torch.cat([latent_dis,#[:, :, :self.latent_dim],
                                              latent_move,#[:, :, :self.latent_dim],                                          
                                              ], dim=2)
                    
                    #shape is (bs*n_agents,latent_dim)
                    latent_view=latent_move.view(bs * self.n_agents, -1)
                    
                    #shape is (bs*n_agents,1)
                    mi = torch.clamp(gaussian_embed.log_prob(latent_view)
                                    +13.9, min=-13.9).sum(dim=1,keepdim=True) / self.latent_dim

                    dissimilarity=self.dis_net(latent_dis_pair.view(-1, 2 * self.latent_dim))
                    #shape is (bs*n_ag,1)
                    dissimilarity = torch.abs(dissimilarity)

                    if dissimilarity_cat is None:
                        
                        dissimilarity_cat = dissimilarity.view(bs, -1).clone()
                    else:
                        
                        dissimilarity_cat = torch.cat([dissimilarity_cat, dissimilarity.view(bs, -1)], dim=1)
                        
                    if mi_cat is None:
                        mi_cat = mi.view(bs, -1).clone()
                    else:
                        mi_cat = torch.cat([mi_cat,mi.view(bs,-1)],dim=1)
            
                #the shape of dissimilarity_cat is (bs,n_agents*n_agents) -> diss with each agents couple
                #the shape of mi_cat is (bs,n_agents*n_agents)
            mi_min=mi_cat.min(dim=1,keepdim=True)[0]
            mi_max=mi_cat.max(dim=1,keepdim=True)[0]
            di_min = dissimilarity_cat.min(dim=1, keepdim=True)[0]
            di_max = dissimilarity_cat.max(dim=1, keepdim=True)[0]
            mi_cat=(mi_cat-mi_min)/(mi_max-mi_min+ 1e-12 )
            #normalized, shape is still (bs,n_agents*n_agents)
            dissimilarity_cat=(dissimilarity_cat-di_min)/(di_max-di_min+ 1e-12 )
            #we want the agents to behave differently
            dis_loss = - torch.clamp(mi_cat+dissimilarity_cat, max=1.0).sum()/bs/self.n_agents           
            dis_norm = torch.norm(dissimilarity_cat, p=1, dim=1).sum() / bs / self.n_agents

            c_dis_loss = (dis_loss + dis_norm) / self.n_agents

            #-----------------------loss 2 done------------------------
            
            #ce_loss has been weighted before with the entropy and kl term
            loss = ce_loss +  c_dis_loss*self.dis_loss_weight
            
            #useless
            self.mi = mi_cat[0]
            self.dissimilarity = dissimilarity_cat[0]


            #------------------LOCAL UTILITY NETWORK (RNN AGENT)-------------------
        
        #now we do it even if we are at evaluation time !!!
        #decoding the roles to condition the agent, 
        #the shape is (bs*n_agents,hidden_size)
        latent = self.latent_net(latent)
        
        fc2_w = self.fc2_w_nn(latent)
        fc2_b = self.fc2_b_nn(latent)
        fc2_w = fc2_w.reshape(-1, self.rnn_hidden_dim, self.n_actions)
        fc2_b = fc2_b.reshape((-1, 1, self.n_actions))

        x = F.relu(self.fc1(inputs))  
        
        x = x.unsqueeze(1)
        #h_in (GRU_l,bs*n_agent,hidden_dim)
        #h_in.cpu()
        h_in = (h_in).reshape(self.GRU_num_layers,-1, self.rnn_hidden_dim).to(self.device)
        #print('x shape', x.shape)
        #x = x.reshape(bs,self.n_agents,-1)
        #h_in = h_in.reshape(self.n_agents,bs,-1)
        _,h_out= self.rnn(x, h_in)
        

        #h_in (bs*n_agent,1,hidden_dim)
        h_out = torch.squeeze(h_out).unsqueeze(1)
        q = torch.bmm(h_out, fc2_w) + fc2_b
        h_out = torch.squeeze(h_out)
        
        return q.view(-1,self.n_agents, self.n_actions),\
               h_out.view(-1,self.n_agents,self.rnn_hidden_dim),\
               loss, c_dis_loss,ce_loss

    def greedy_action_id(self,inputs,hs):
        hs.to(self.device)
        qvals, h,_,_,_ = self.forward(inputs,hs,train_mode=False)

        action_idx = torch.argmax(qvals,dim=-1)
        
        return action_idx, h


if __name__ == '__main__':
    
    agent = RomaAgent(input_shape=64+5,n_agents=3,n_actions=16,latent_dim=8,rnn_hidden_dim=32,
                        batch_size=2,fc_hidden_size=12)
    inputs = torch.rand((agent.batch_size,agent.n_agents,agent.input_shape))
    print(inputs.shape)
    h_in = agent.fc1.weight.new(agent.batch_size,agent.n_agents, agent.rnn_hidden_dim).zero_()
    agent.forward(inputs,h_in)
    
    print('daje brah')


    





    ##it generates mean and variances from which the role decoder will sample
    #self.role_encoder = nn.Sequential(nn.Linear(input_shape,hidden_dim),
    #                                  activation,
    #                                 nn.Linear(hidden_dim,latent_dim*2)  )
#
    ##sample from the role encoder and we get a dimension which is latent_dim 
#
    #self.inference_net = nn.Sequential(nn.Linear(rnn_hidden_dim+input_shape,hidden_dim),
    #                                    activation,
    #                                   nn.Linear(hidden_dim,latent_dim*2))
#
    ##THEN WE DO KL OF A SAMPLE SAMPLED FROM THE
    ##ROLE DECODER AND INFERENCE NET TO MAKE 
    ##SURE THAT THE TRAJECTORIES WON'T CHANGE TOO FAST
    #
    #self.role_decoder = nn.Sequnrtial(nn.Linear(),
    #                                activation,
    #                                nn.linear(fc_hidden_size,))
#
    #self.diss_net=nn.Sequential(nn.Linear(rnn_hidden_dim*2,hidden_dim),
    #                                    activation,
    #                                    nn.Linear(hidden_dim,1))