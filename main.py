import argparse
import random
import numpy as np
from gym_derk.envs import DerkEnv
#from agent import Agents

#USE THIS TO TRAIN 2 ROMA AGENTS
#from roma_agent import Agents

from agents_brain import Agents

from utils.environment_setting import CustomEnvironment

def evaluate(env=None, n_episodes=1):
    
    custom_envirment = CustomEnvironment(training_mode=False)
    agent = Agents(custom_envirment,'RVsR')
    

    agent.evaluation()



def train():
    custom_envirment = CustomEnvironment(training_mode=True)
    agent = Agents(custom_envirment,'RVsR')
    agent.agent_1.load()
    agent.agent_2.load()
    agent.train()
    
    

def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()

    if args.evaluate:
        evaluate()

    
if __name__ == '__main__':
    main()