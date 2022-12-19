import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Poiché qui viene utilizzato RNN, l'ultimo hidden_state è necessario ogni volta.
Per i dati di un episodio, ogni osservatore ha bisogno dell'ultimo hidden_state per scegliere un'azione.
Pertanto, non è possibile estrarre direttamente in modo casuale un batch di esperienza e inserirlo nella rete 
neurale, pertanto qui è necessario un batch di episodi e la transizione della stessa posizione 
di questo batch di episodi viene trasmessa ogni volta.
In questo modo, lo stato_nascosto può essere salvato e la prossima esperienza 
sarà passata la volta successiva
'''

class RNNAgent(nn.Module):
    def __init__(self,n_agents,n_actions, input_shape, rnn_hidden_dim,batch_size ):
        super(RNNAgent, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device="cpu"
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.rnn_hidden_dim=rnn_hidden_dim
        self.batch_size = batch_size
        self.GRU_num_layers=1

        self.fc1 = nn.Linear(self.input_shape,  self.rnn_hidden_dim) 

        self.rnn = nn.GRU( self.rnn_hidden_dim, self.rnn_hidden_dim,
                        num_layers=self.GRU_num_layers,batch_first=True)

        self.fc2 = nn.Linear( self.rnn_hidden_dim, self.n_actions) 
        self.hidden_state = self.fc1.weight.new(self.batch_size,self.n_agents, self.rnn_hidden_dim).zero_().to(self.device)

        

    def forward(self, inputs, hidden_state):
        
        inputs = inputs.reshape(-1,self.input_shape)
        h_in = hidden_state.reshape(self.GRU_num_layers,-1, self.rnn_hidden_dim).to(self.device)

        x = F.relu(self.fc1(inputs))
        
        x = x.unsqueeze(1)
        
        y,h_out = self.rnn(x, h_in)  
        q = self.fc2(h_out)
        
        
        return q.view(-1,self.n_agents, self.n_actions),\
               h_out.view(-1,self.n_agents,self.rnn_hidden_dim) #I thin q shoud be some vector like  [action]
    
    def greedy_action_id(self,inputs,hs):
        hs.to(self.device)
        qvals, h= self.forward(inputs,hs)
        action_idx = torch.argmax(qvals,dim=-1)
        return action_idx, h
    
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret


if __name__ == '__main__':
    
    agent = RNNAgent(input_shape=64+5,n_agents=3,n_actions=16,rnn_hidden_dim=32,
                        batch_size=2)
    inputs = torch.rand((agent.batch_size,agent.n_agents,agent.input_shape))
    
    h_in = agent.fc1.weight.new(agent.batch_size,agent.n_agents, agent.rnn_hidden_dim).zero_()
    q,h = agent.forward(inputs,h_in)

    print('q',q.shape)
    print('h',h.shape)
    
    print('daje brah')

