from gym_derk.envs import DerkEnv

env = DerkEnv()

for t in range(3):
  observation_n = env.reset()
  cnt = 0
  while True:
    action_n = [env.action_space.sample() for i in range(env.n_agents)]
    observation_n, reward_n, done_n, info = env.step(action_n)
    if all(done_n):
      print("Episode finished")
      break
    
    print(env.team_stats)

    cnt +=1
env.close()