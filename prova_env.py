from gym_derk.envs import DerkEnv
import numpy as np

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
    observation_n, reward_n, done_n, info = env.step(action_n)
    globs = global_state(observation_n)
   # print("observation number",cnt)
   # print(globs)
    
    #if cnt==3:
    #  gvrfgv
    cnt +=1
env.close()