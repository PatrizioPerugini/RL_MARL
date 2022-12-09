from gym_derk.envs import DerkEnv
from wrappers import Discrete_actions_space

env = DerkEnv()
#action n is a [num_agent,num_action] matrix,
#each row corresponds to the action taken by that agent and it's a list of dim 5, where:
#[MoveX (-1,1) , Rotate (-1,1), ChaseFocus(0,1), CastingSlot(int), Change_focus(int)]
discrete_actions = Discrete_actions_space(3,3,3)




while True:
    observation_n = env.reset()
    cnt = 0
   
    while True:
        action_n = [discrete_actions.sample() for i in range(env.n_agents)]
        observation_n, reward_n, done_n, info = env.step(action_n)
        if all(done_n):
            print("Episode finished")
        break
    cnt += 1
env.close()