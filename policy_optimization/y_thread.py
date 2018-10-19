from __future__ import print_function

import numpy as np
import tensorflow as tf
from ppo import PPOTrain
from policy_net import Policy_net
import threading
import time
import gym
import operator
import itertools

thread_number = 30

score_list = []
sess = tf.InteractiveSession()
Policy = Policy_net(sess, 'policy')
Old_Policy = Policy_net(sess, 'old_policy')
PPO = PPOTrain(sess, Policy, Old_Policy)
sess.run(tf.global_variables_initializer())

def add(number):
    env = gym.make('CartPole-v0')
    done = False
    observations = []
    actions = []
    v_preds = []
    rewards = []
    state = env.reset()
    global_step = 0
    while not done:
        global_step += 1
        action, value = Policy.act(state)
        next_state, reward, done, _ = env.step(action)

        if done: reward = -1

        observations.append(state)
        actions.append(action)
        v_preds.append(value)
        rewards.append(reward)

        state = next_state
    
    v_preds_next = v_preds[1:] + [0]
    gaes = PPO.get_gaes(rewards, v_preds, v_preds_next)
    observations = np.array(observations)
    actions = np.array(actions).astype(dtype=np.int32)
    rewards = np.array(rewards).astype(dtype=np.float32)
    v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
    gaes = np.array(gaes).astype(dtype=np.float32)

    list_observations.append(observations)
    list_actions.append(actions)
    list_rewards.append(rewards)
    list_v_preds_next.append(v_preds_next)
    list_gaes.append(gaes)
    
    env.close()

for episodes in range(500):
    list_observations = []
    list_actions = []
    list_v_preds = []
    list_rewards = []
    list_gaes = []
    list_v_preds_next = []
    threads = [threading.Thread(target=add, args=(i, )) for i in range(thread_number)]
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    list_observations = np.array(list(itertools.chain.from_iterable(list_observations)))
    list_actions = np.array(list(itertools.chain.from_iterable(list_actions)))
    list_rewards = np.array(list(itertools.chain.from_iterable(list_rewards)))
    list_v_preds_next = np.array(list(itertools.chain.from_iterable(list_v_preds_next)))
    list_gaes = np.array(list(itertools.chain.from_iterable(list_gaes)))

    PPO.assign_policy_parameters()
    inp = [list_observations, list_actions, list_rewards, list_v_preds_next, list_gaes]
    for epoch in range(4*thread_number):
        sample_indices = np.random.randint(low=0, high=list_observations.shape[0], size=64)  # indices are in [low, high)
        sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
        PPO.train(obs=sampled_inp[0],
                  actions=sampled_inp[1],
                  rewards=sampled_inp[2],
                  v_preds_next=sampled_inp[3],
                  gaes=sampled_inp[4])
    print(episodes, sum(list_rewards)/thread_number)