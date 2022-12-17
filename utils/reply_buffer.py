import numpy as np
import threading
from gym_derk.envs import DerkEnv



class ReplayBuffer:
    def __init__(self,n_agents,state_shape,obs_shape,buffer_size,episode_limit):
        
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.obs_shape =obs_shape
        
        self.size = buffer_size
        self.episode_limit =episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info.
        # Il significato di ciascuna dimensione del buffer: 
        # 1——il numero dell'episodio 
        # 2——il numero della transizione nell'episodio 
        # 3——il numero dei dati dell'agente 
        # 4——dimensione specifica dell'oss
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape[1]]),
                        'a': np.empty([self.size, self.episode_limit, self.n_agents]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape[0],self.state_shape[1]]),  
                        'r': np.empty([self.size, self.episode_limit, self.n_agents]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape[1]]),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape[0],self.state_shape[1]]),
                        'terminated': np.empty([self.size, self.episode_limit, 1]),
                        'episode_len':np.empty([self.size])
                        }

    def get_capacity(self):
        if self.current_idx<self.size:
            return round(self.current_idx/self.size,2)
        else:
            return 1.0
        # thread lock
    #   self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        #with self.lock:
        #_ = self._get_storage_idx(inc=batch_size)
        
        idxs = (self.current_idx) % self.size
        
        # store the informations
        self.buffers['o'][idxs] = episode_batch['o']
        self.buffers['a'][idxs] = episode_batch['a']
        self.buffers['s'][idxs] = episode_batch['s']
        self.buffers['r'][idxs] = episode_batch['r']
        self.buffers['o_next'][idxs] = episode_batch['o_next']
        self.buffers['s_next'][idxs] = episode_batch['s_next']
        self.buffers['terminated'][idxs] = episode_batch['terminated']
        self.buffers['episode_len'][idxs] = episode_batch['episode_len']
        self.current_idx += 1

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_idx, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        # Memorizza le ultime self.size esperienze nel buffer 
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)#get sequences of batches
            self.current_idx += inc #increase the buffer dim
        #we should start from the beginning once we are at the end 
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx) 
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])#get seq of batches from and and start
            self.current_idx = overflow
        #we are at the end, start from scratch
        else:
            idx = np.arange(0, inc)#get seq of batches from start
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

if __name__ == '__main__':
    #def __init__(self, n_actions,n_agents,state_shape,obs_shape,buffer_size,episode_limit):
    env = DerkEnv()
    observation_n = env.reset()
    
    obs_shapee=observation_n.shape
    
    #input()
    rb = ReplayBuffer(n_actions=15,
                    n_agents=6,
                    state_shape=obs_shapee,
                    obs_shape=obs_shapee,
                    buffer_size=50,
                    episode_limit=16)

    print(rb.buffers['o'].shape)
    print(rb.buffers['a'][0][0][0].shape)
    print("WELL HELLOO")

