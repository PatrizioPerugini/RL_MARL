import argparse
import random
import numpy as np
from gym_derk.envs import DerkEnv
#from agent import Agents
from roma_agent import Agents
from utils.environment_setting import CustomEnvironment

def evaluate(env=None, n_episodes=1):
    env = CustomEnvironment()
    agent = Agents()
    
    for episode in range(n_episodes):
        observation_n = env.reset()
        total_rewards = [0,0]
        done_n = [False]
        last_action_n = np.zeros((6,5))

        while True:     
            input_rnn = np.hstack((observation_n,last_action_n))

            action_1 = agent.agent_1.act(input_rnn)
            action_2 = agent.agent_2.act(input_rnn)
            
            action_n = action_1 + action_2
            
            observation_n, reward_n, done_n, info = env.step(action_n)

            last_action_n = action_n
            total_rewards[0] += sum(reward_n[0:2])
            total_rewards[1] += sum(reward_n[3:5])
            if done_n[0]: break
        
        total_rewards.append(total_rewards)
    
    print('Mean Reward:', np.mean(total_rewards))


def train():
    custom_envirment = CustomEnvironment()
    agent = Agents(custom_envirment)
    #agent.roll_in_episode()

    #TODO
    agent.train()
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