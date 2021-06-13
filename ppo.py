from pong_constants import Constants
import gym
import time
import numpy as np
from utils import states_to_prob, \
        convert_immediate_rewards_to_discounted_rewards, \
        convert_immediate_rewards_to_discounted_rewards_optimized, \
        normalize_future_rewards
from env_utils.pong_utils import collect_trajectories, play, RIGHT, LEFT
import torch
from parallelEnv import parallelEnv
import numpy as np
import progressbar as pb
from config import config
from models.pong_simple import Policy
import torch.optim as optim
import uuid


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount = 0.995, epsilon=0.1, beta=0.01):

    rewards = np.array(rewards)
    
    '''
    calculate discounted rewards from immediate rewards
    '''
    discounted_rewards = []
    for i in range(len(rewards[0])):
        discounted_rewards.append(convert_immediate_rewards_to_discounted_rewards(rewards[:,i], gamma=discount))
    discounted_rewards = np.array(discounted_rewards)
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = discounted_rewards.permute(1, 0).float().to(device)

    
    '''
    Normalize rewards
    '''
    discounted_rewards = normalize_future_rewards(discounted_rewards.tolist())
    discounted_rewards = torch.tensor(discounted_rewards).float().to(device)
                                  

    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs).to(device)
    
    '''
    get new probabilites to the same states
    '''
    
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)
    new_probs = new_probs.to(device)

    '''
    IMPORTANCE SAMPLING:
    calculate ratio of old_probs and new_probs, clipped ratio with a fixed epsilon
    pick min ( ratio * reward, clipped_ratio* reward) for every element in the list
    '''
    r = (new_probs/old_probs)
    clip = torch.clamp(r, 1-epsilon, 1+epsilon)
    r = torch.min(r * discounted_rewards, clip * discounted_rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # prevents policy to become exactly 0 or 1 helps exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    return torch.mean(r  + beta*entropy)


def train():
    experiment_id = uuid.uuid4()

    # PongDeterministic does not contain random frameskip
    # so is faster to train than the vanilla Pong-v4 environment
    env = gym.make('PongDeterministic-v4')
    print("List of available actions: ", env.unwrapped.get_action_meanings())
    # List of available actions:  ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

    '''
    initialize model
    '''
    policy=Policy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config['lr'])


    num_epsiodes = config['num_epsiodes']

    # widget bar to display progress

    widget = ['training loop: ', pb.Percentage(), ' ', 
            pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=num_epsiodes).start()


    envs = parallelEnv('PongDeterministic-v4', n=config['num_parallel_envs'], seed=1234)
    epsilon = config['epsilon']
    beta = config['beta']
    tmax = config['tmax']
    SGD_epoch = config['SGD_epoch']

    # keep track of progress
    mean_rewards = []

    for e in range(num_epsiodes):

        # collect trajectories
        old_probs, states, actions, rewards = \
            collect_trajectories(envs, policy, tmax=tmax)
            
        total_rewards = np.sum(rewards, axis=0)


        # gradient ascent step
        for _ in range(SGD_epoch):
            
            L = -clipped_surrogate(policy, old_probs, states, actions, rewards,\
                                            epsilon=epsilon, beta=beta)
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L
            
        
        # the clipping parameter reduces as time goes on
        epsilon*=.999
        
        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.995
        
        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))
        
        # display some progress every 20 iterations
        if (e+1)%20 ==0 :
            print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
            print(total_rewards)
            
        # update progress widget bar
        timer.update(e+1)
        
        timer.finish()

        #torch.save(policy, experiment_id)
