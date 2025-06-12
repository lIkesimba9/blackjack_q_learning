import numpy as np
import itertools
import gymnasium
from typing import Dict





def get_observation_range_from_discrete(obseration: gymnasium.spaces.Discrete):
    return np.arange(obseration.n)

class BlackJackAgent:
    
    def __init__(self, env, count_action: int, get_action_func):

        self.states = self.get_all_states_from_env(env)
        self.q_policy = dict(zip(self.states, np.zeros((len(self.states), count_action))))
        self.env = env
        self.get_action_func = get_action_func
        
    
    def get_all_states_from_env(self, env):
        states = [get_observation_range_from_discrete(obseration) for obseration in env.observation_space]
        return list(itertools.product(*states))
    
    def q_learning(self, num_episodes: int,epsilon: float, alpha: float, gamma: float):
        
        for _ in range(num_episodes):
            state, info = self.env.reset()
            terminated = False
            while not terminated:
                action = self.get_action_func(self.q_policy, state, epsilon, self.env)
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                self.q_policy[state][action] = self.q_policy[state][action] + alpha * (reward + gamma * max(self.q_policy[observation]) - self.q_policy[state][action])
                state = observation
    
    def get_strategy(self):
        return dict(zip(self.states, np.argmax(list(self.q_policy.values()), axis=1)))