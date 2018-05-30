from ppo import PPO
import gym
import tensorflow as tf
import numpy as np
import time

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.001
C_LR = 0.002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]

env = gym.make('Pendulum-v0').unwrapped

cr_net_shape = [3,100,1]
cr_act_shape = [tf.nn.relu, tf.nn.relu]
ac_net_shape = [3,100, 100, 1]
ac_act_shape = [tf.nn.relu, tf.nn.relu]
act_range = 2

ppo = PPO(act_range, METHOD, A_LR, C_LR)
all_ep_r = []
saver = tf.train.Saver()

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        #if ep % 10 == 0:
        #    env.render()
        #    time.sleep(0.01)
        a = ppo.choose_action(s, act_range)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br, METHOD, A_UPDATE_STEPS, C_UPDATE_STEPS)
            #saver.save(ppo.sess, './ppo.ckpt')

    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )