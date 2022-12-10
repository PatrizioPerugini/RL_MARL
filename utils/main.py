import argparse
import random
import numpy as np
from gym_derk.envs import DerkEnv
from wrappers import Discrete_actions_space
from agent import Agent
from prova_buffer import ReplayBuffer

def evaluate(env=None, n_episodes=1):
    env = DerkEnv()
    agent_1 = Agent(env,team = 1)
    agent_2 = Agent(env,team = 2)
    
    for episode in range(n_episodes):
        observation_n = env.reset()
        total_rewards = [0,0]
        done_n = [False]

        while True:     
            action_1 = agent_1.act(observation_n)
            action_2 = agent_2.act(observation_n)
            
            action_n = action_1 + action_2
            
            observation_n, reward_n, done_n, info = env.step(action_n)

            total_rewards[0] += sum(reward_n[0:2])
            total_rewards[1] += sum(reward_n[3:5])
            if done_n[0]: break
        
        total_rewards.append(total_rewards)
    
    print('Mean Reward:', np.mean(total_rewards))


def train():
    

    #TODO
    #agent.train()
    pass
    

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