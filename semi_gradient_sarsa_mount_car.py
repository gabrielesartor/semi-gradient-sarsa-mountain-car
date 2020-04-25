import sys
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import pickle

import gym
from gym import spaces
from gym.utils import seeding

"""
Implementation of semi-gradient sarsa method with state aggregation representation
for the mountain car's Gym environment.
"""

#Simulation parameters
NUM_EPISODES = 100000
MAX_T = 200
ALPHA = 0.01
GAMMA = 0.98
EPSILON = 0.2

#Test flgas
DEBUG = False
RENDER_POLICY = True
MEAN_RANGE = 10
NUM_EPISODES_PLOT = 100
ALGORITHM = "sarsa"


def get_state(env):
    """
    It calculates the vector representation of the current state. In this case,
    it is used a state aggregation representation.
    """
    segmentation_factor = 2000
    pos_segment = (env.high[0] - env.low[0]) / segmentation_factor
    vel_segment = (env.high[1] - env.low[1]) / segmentation_factor
    state = env.state
    coarse_state = np.zeros(2*segmentation_factor)


    coarse_state[int((state[0] - env.low[0])/ pos_segment)] = 1

    coarse_state[int((state[1] - env.low[1])/ vel_segment) + segmentation_factor] = 1

    return coarse_state

def value_approx_slow(env, state, action, weights):
    """
    It calculates the value of the state-action pair multiplying the state-action
    pair by the learned weights (without using numpy.dot).
    """
    action_one_hot_vector = np.zeros(env.action_space.n)
    action_one_hot_vector[action] = 1
    approximation = 0

    w_i = 0
    for s_i in state:
        for a_i in action_one_hot_vector:
            approximation = approximation + (s_i*a_i*weights[w_i])
            w_i = w_i + 1
    return approximation

def value_approx(env, state, action, weights):
    """
    It calculates the value of the state-action pair multiplying the state-action
    pair by the learned weights.
    """
    action_one_hot_vector = np.zeros(env.action_space.n)
    action_one_hot_vector[action] = 1

    s_a = np.zeros(len(weights))

    w_i = 0
    for s_i in state:
        for a_i in action_one_hot_vector:
            s_a[w_i] = s_i*a_i
            w_i = w_i + 1

    return np.dot(s_a, weights)


def value_approx_grad_slow(env, state, action, weights):
    """
    It calculates the gradient of the state-action pair (using list).
    """
    action_one_hot_vector = np.zeros(env.action_space.n)
    action_one_hot_vector[action] = 1
    gradient = []

    w_i = 0
    for s_i in state:
        for a_i in action_one_hot_vector:
            df = s_i*a_i
            gradient.append(df)
            w_i = w_i + 1
    return np.array(gradient)

def value_approx_grad(env, state, action, weights):
    """
    It calculates the gradient of the state-action pair.
    """
    action_one_hot_vector = np.zeros(env.action_space.n)
    action_one_hot_vector[action] = 1
    gradient = np.zeros(len(weights))

    w_i = 0
    for s_i in state:
        for a_i in action_one_hot_vector:
            gradient[w_i] = s_i*a_i
            w_i = w_i + 1

    return gradient

def select_action_e_greedy(env, Q_values, epsilon = EPSILON):
    """
    It selects a random action with probability epsilon, otherwise the best learned until now.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    else:
        return np.argmax(Q_values)

def learning_episode(env, weights, algorithm = "sarsa"):
    """
    It calculates a whole simulation episode using the algorithm passed as argument.
    """
    total_reward = 0

    s = get_state(env)
    Q_values = [value_approx(env, s, action, weights) for action in range(env.action_space.n)]
    a = select_action_e_greedy(env, Q_values)
    Q_prev = Q_values[a]

    for t in range(MAX_T):

        _, reward, done, _ = env.step(a)

        total_reward = total_reward + reward

        if env.state[0]>0.5:
            weights = weights + ALPHA*(reward - Q_prev) * value_approx_grad(env, s, a, weights)
            break

        s_new = get_state(env)
        Q_values_s_new = [value_approx(env, s_new, action, weights) for action in range(env.action_space.n)]
        a_next = select_action_e_greedy(env, Q_values_s_new)
        Q_next_a = Q_values_s_new[a_next]

        #updata rule
        if algorithm == "sarsa":
            weights = weights + ALPHA*(reward + GAMMA*Q_next_a - Q_prev) * value_approx_grad(env, s, a, weights)
            # weights = weights / np.linalg.norm(weights)
        else:
            raise NameError('Unknown algorithm name!')

        if DEBUG:
            env.render()
            print("Q_values:", Q_values)
            print("action: ", a)
            print("s_new: ", s_new)
            print("Q_prev: ", Q_prev)
            print("Q_next_a: ", Q_next_a)
            print("env.state: ", env.state)
            print("reward :", reward)

        s = s_new
        a = a_next
        Q_values = Q_values_s_new
        Q_prev = Q_next_a

    return total_reward, weights

def training(env, weights, rewards, algorithm):
    """
    It carries out the training phase executing NUM_EPISODES of trials in the environment.
    """
    for episode in range(NUM_EPISODES):
        env.reset()

        time_start = time.time()
        episode_reward, weights = learning_episode(env, weights, algorithm)
        print("episode time: ", time.time()-time_start)

        rewards[episode] = episode_reward

        if episode % NUM_EPISODES_PLOT == 0 and episode!=0:
            plt.plot(range(episode+1), rewards[:episode+1], "b")
            plt.axis([0, episode, np.min(rewards[:episode+1]), np.max(rewards[:episode+1])])
            plt.pause(0.1)

            if RENDER_POLICY:
                render_policy(env, weights)
    return weights

def render_policy(env, weights, epsilon = 0):
    """
    It shows the current learned behaviour on the GUI
    """
    env.reset()

    for t in range(MAX_T):
        env.render()

        s = get_state(env)
        # print("weights inside render: ", weights)
        Q_values = [value_approx(env, s, action, weights) for action in range(env.action_space.n)]
        # print("Q_values inside render: ", Q_values)
        a = select_action_e_greedy(env, Q_values, epsilon)
        # print("a:", a)

        s_new, reward, done, _ = env.step(a)

        if env.state[0]>0.5:
            print("I've reached the goal!")
            break

    print("Policy executed.")
    env.render()

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")

    env.reset()
    env_dim = len(get_state(env))
    rewards = np.zeros(NUM_EPISODES)

    weights = np.zeros(env_dim*env.action_space.n)

    weights = training(env, weights, rewards, algorithm = ALGORITHM)

    print("Execute final policy...")
    render_policy(env, weights)
    print("Everything is done!")

    env.close()
    plt.show()
