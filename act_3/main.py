import argparse
import random
import numpy as np
from gym_derk.envs import DerkEnv
#from agent import Agents

#USE THIS TO TRAIN 2 ROMA AGENTS
#from roma_agent import Agents

from agents_brain import Agents

from utils.environment_setting import CustomEnvironment

def evaluate():
    
    custom_envirment = CustomEnvironment(training_mode=False)
    team1 = str(input('Initial of the first team (maiusc):'))
    team2 = str(input('Initial of the second team (maiusc):'))
    match = team1+'vs'+team2
    agent = Agents(custom_envirment,match)
    agent.agent_1.load()
    agent.agent_2.load()

    agent.evaluation()



def train():
    custom_envirment = CustomEnvironment(training_mode=True)
    agent = Agents(custom_envirment,'RvsR')
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