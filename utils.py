import numpy as np
import torch

def convert_immediate_rewards_to_discounted_rewards(immediate_rewards_list, gamma=0.995):
    
    '''
    immediate_rewards_list is a 1d list
    ex: [1, 4, 0 , 0, 10]

    Turn  this into discounted rewards like below:
    [15, 14, 10, 10, 10]
    i.e. [1(gamma^0) + 4(gamma^1) + 0(gamma^2) + 0(gamma^3) +10(gamma^4), 
          4(gamma^0) + 0(gamma^1) + 0(gamma^2) + 10(gamma^3), 
          0(gamma^0) + 0(gamma^1) + 10(gamma^2), 
          0(gamma^0) + 10(gamma^1), 
          10(gamma^0)]
    '''

    discounted_rewards = []
    for i in range(len(immediate_rewards_list)):
        exponent = np.arange(len(immediate_rewards_list[i:])) #[0, 1, 2 ...len(immediate_rewards)]
        discounts = gamma ** exponent #gamma ** [0, 1, 2, ...] ==> [POW(gamma, 0), POW(gamma, 1) ... ]
        discounted_reward = np.sum(np.multiply(immediate_rewards_list[i:], discounts))
        discounted_rewards.append(discounted_reward)
    return discounted_rewards

def convert_immediate_rewards_to_discounted_rewards_optimized(immediate_rewards_list, gamma=0.995):

    '''
    immediate_rewards_list is a 1d list
    ex: [1, 4, 0 , 0, 10]

    Turn  this into discounted rewards like below:
    [15, 14, 10, 10, 10]
    i.e. [1(gamma^0) + 4(gamma^1) + 0(gamma^2) + 0(gamma^3) +10(gamma^4), 
          4(gamma^0) + 0(gamma^1) + 0(gamma^2) + 10(gamma^3), 
          0(gamma^0) + 0(gamma^1) + 10(gamma^2), 
          0(gamma^0) + 10(gamma^1), 
          10(gamma^0)]
    '''

    
    immediate_rewards_list = np.array(immediate_rewards_list)
    discounts = gamma**np.arange(len(immediate_rewards_list))
    discounted_rewards =  immediate_rewards_list * discounts[np.newaxis]

    '''
    reverse discounted rewards, 
    do cumulative sum to add rewards to get discounted rewards, reverse again
    '''
    return discounted_rewards[::-1].cumsum(axis=0)[::-1]

def normalize_future_rewards(rewards_future):

    '''
    rewards_future is a numpy array of size  t_max X n_envs 
    (
    If parallel envs are used, 
    t_max is the number of steps in the episode
    n_envs is the number of parallel environments
    )

    '''
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    return rewards_normalized

def states_to_prob(policy, states):
    states = torch.stack(states)
    policy_input = states.view(-1,*states.shape[-3:])
    return policy(policy_input).view(states.shape[:-3])
