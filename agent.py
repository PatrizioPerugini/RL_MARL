import torch
from wrappers import Discrete_actions_space
from gym_derk.envs import DerkEnv



class Agent():

    def __init__(self,env,team):#,replay_buffer):
        super(Agent, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.discrete_actions = Discrete_actions_space(3,3,3)
        self.team = team
        self.n_agents = env.n_agents / 2    #teams have the same number of components
        self.team_members_id = self.team_split([i for i in range(6)])
        
        # Training stuff
        self.tot_episodes = 500
        self.max_steps_per_episode = 1000
        self.gamma=0.99
        self.learning_rate=0.00025
        self.epsilon=0.7
        self.min_epsilon = 0.2
        self.init_epsilon = self.epsilon
        self.epsilon_decay = 0.95
        self.batch_size = 64      

        


    def team_split(self,obj):
        if self.team ==1:
            return obj[0:3]
        else:
            return obj[3:6]
    
    def act(self,observation_n):
        obs = self.team_split(observation_n)
        actions = [ self.discrete_actions.sample() for i in self.team_members_id]
        return actions
