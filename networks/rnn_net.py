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
    def __init__(self, input_shape, rnn_hidden_dim, num_actions):
        super(RNNAgent, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.rnn_hidden_dim=rnn_hidden_dim
        #the output dim of this layer should match the dimension of the hidden state
        self.fc1 = nn.Linear(input_shape,  self.rnn_hidden_dim) 
        #process the new input-trajectory pair
        self.rnn = nn.GRUCell( self.rnn_hidden_dim, self.rnn_hidden_dim)
        #finally compute the new q 
        self.fc2 = nn.Linear( self.rnn_hidden_dim, self.num_actions) #there are 5 actions
        self.hidden_state = self.init_hidden()
        

    def init_hidden(self,batch_size=1):
        # make hidden states on same device as model
        
        return self.fc1.weight.new(batch_size, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        #update gate and reset gate need these informations to be computed, the output coincide 
        #with the new hidden state which will be fed to the last layer to produes the 
        #actual q function
        h = self.rnn(x, h_in)  
        q = self.fc2(h)
        #q function (take the argmax) + new hidden state (will be given as input for the next GRU)
        #in the paper q is actually Q(traj,action) once the epsilon-greedy is done
        return q, h #I thin q shoud be some vector like  [action]
    
    def greedy_action_id(self,inputs):
        qvals, h = self.forward(inputs,self.hidden_state)
        self.hidden_state = h
        action_idx = torch.argmax(qvals).item()
        return action_idx, self.hidden_state


