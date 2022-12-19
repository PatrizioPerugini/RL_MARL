from gym_derk.envs import DerkEnv
from wrappers import Discrete_actions_space
import random
from environment_setting import CustomEnvironment

environment_settings = CustomEnvironment()
env = environment_settings.env


#action n is a [num_agent,num_action] matrix,
#each row corresponds to the action taken by that agent and it's a list of dim 5, where:
#[MoveX (-1,1) , Rotate (-1,1), ChaseFocus(0,1), CastingSlot(int), Change_focus(int)]
discrete_actions = Discrete_actions_space(0.5,0.5)



for t in range(3):
    observation_n = env.reset()
    done = False
   
    while not done:
        action_1 = discrete_actions.sample() 
        action_2 = discrete_actions.sample() 
        action_n = action_1[1]+action_2[1]
        observation_n, reward_n, done_n, info = env.step(action_n)
        done = done_n[0]
        
env.close()








'''
def create_env():
    home_team=[
      { 'primaryColor': '#FF0000','secondaryColor': '#FFA500' },
      { 'primaryColor': '#FF0000','secondaryColor': '#FFA500' },
      { 'primaryColor': '#FF0000','secondaryColor': '#FFA500' },]
    env = DerkEnv(n_arenas = 2,,
      away_team=[
      { 'primaryColor': '#ADD8E6','secondaryColor': '#FFFFFF'},
      { 'primaryColor': '#ADD8E6','secondaryColor': '#FFFFFF'},
      { 'primaryColor': '#ADD8E6','secondaryColor': '#FFFFFF'}]
)
'''
