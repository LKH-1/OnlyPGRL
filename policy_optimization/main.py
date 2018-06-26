import gym
import numpy as np
import tensorflow as tf
from policy_net import Policy_net
from ppo import PPOTrain

env = gym.make('CartPole-v1')
state_size = 4
action_space = 2

r = tf.placeholder(tf.float32)
rr = tf.summary.scalar('reward', r)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./board/cate', sess.graph)
    tf.set_random_seed(1234)
    Policy = Policy_net('policy', 4, 2)
    Old_Policy = Policy_net('old_policy', 4, 2)
    PPO = PPOTrain(Policy, Old_Policy, gamma=0.95, mode='kl_pen') # mode = 'clip' or 'kl_pen'
    sess.run(tf.global_variables_initializer())
    first_episodes = 0
    for episodes in range(500):
        observations = []
        actions = []
        v_preds = []
        rewards = []
        state = env.reset()
        t = 0
        reward = 0
        done = False
        while not done:
            #if episodes % 10 == 0:
            #    env.render()
            t += 1
            act, v_pred = Policy.act([state], stochastic=True)
            act, v_pred = np.asscalar(act), np.asscalar(v_pred)
            observations.append(state)
            actions.append(act)
            v_preds.append(v_pred)
            rewards.append(reward)
            next_state, reward, done, _ = env.step(act)

            if done:
                v_preds_next = v_preds[1:] + [0]
                reward = -1
            else:
                state = next_state

        gaes = PPO.get_gaes(rewards, v_preds, v_preds_next)
        observations = np.reshape(observations, [-1, 4])
        actions = np.array(actions).astype(dtype=np.int32)
        rewards = np.array(rewards).astype(dtype=np.float32)
        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
        gaes = np.array(gaes).astype(dtype=np.float32)
        gaes = (gaes - gaes.mean()) / gaes.std()

        PPO.assign_policy_parameters()
        inp = [observations, actions, rewards, v_preds_next, gaes]
        for epoch in range(4):
            sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)  # indices are in [low, high)
            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
            PPO.train(obs=sampled_inp[0],
                      actions=sampled_inp[1],
                      rewards=sampled_inp[2],
                      v_preds_next=sampled_inp[3],
                      gaes=sampled_inp[4])
            summary = sess.run(merged, feed_dict={r: t})
            writer.add_summary(summary, episodes)
        if t == 200:
            first_episodes = episodes
        print(episodes, t, first_episodes)