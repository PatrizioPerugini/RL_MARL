from gym_derk.envs import DerkEnv
import numpy as np
import random
from collections import deque , namedtuple
#REPLAY BUFFER IS FORMED BY (S_t,U_t,R_t,S_t+1)
class ReplayMemory:
    def __init__(self, capacity=10):
        self.memory = deque(maxlen=capacity)
  
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size=2):
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones
buffer = ReplayMemory()

env = DerkEnv()

def global_state(observation_n):
  attrib = [28,29,30,22]
  ris = np.zeros((observation_n.shape[0],len(attrib)))
  for i in range(observation_n.shape[0]):
    for j in range(len(attrib)):
      ris[i,j] =(observation_n[i,attrib[j]])
      
  return ris


for t in range(3):
  observation_n = env.reset()
  cnt = 0
  while True:
    action_n = [env.action_space.sample() for i in range(env.n_agents)]
    #each action is then actually a list of 5 actions
    observation_n, reward_n, done_n, info = env.step(action_n)
    #state, action, reward, next_state, done
    buffer.add(observation_n,action_n,reward_n,observation_n,done_n)
    if cnt>2:
      input()
      samples=buffer.sample()
      print(len(samples[0]))
      input()
    #globs = global_state(observation_n)
   # print("observation number",cnt)
   # print(globs)
    cnt +=1
env.close()