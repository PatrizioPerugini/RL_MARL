from gym_derk.envs import DerkEnv
from wrappers import Discrete_actions_space

env = DerkEnv()
#action n is a [num_agent,num_action] matrix,
#each row corresponds to the action taken by that agent and it's a list of dim 5, where:
#[MoveX (-1,1) , Rotate (-1,1), ChaseFocus(0,1), CastingSlot(int), Change_focus(int)]
discrete_actions = Discrete_actions_space(0.1,0.1,3)
print(discrete_actions.count)



for t in range(3):
    observation_n = env.reset()
    done = False
   
    while not done:
        action_n = [discrete_actions.sample() for i in range(env.n_agents)]
        
        observation_n, reward_n, done_n, info = env.step(action_n)
        done = done_n[0]
        
env.close()