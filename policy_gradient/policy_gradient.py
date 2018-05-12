import tensorflow as tf
import numpy as np
import gym

class PG:
    def __init__(self, sess, state_size, action_size):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size

        self.X = tf.placeholder(tf.float32, [None, self.state_size])
        self.advantages = tf.placeholder(tf.float32, [None, 1])
        self.Y = tf.placeholder(tf.float32, [None, self.action_size])
        self.r = tf.placeholder(tf.float32, [None, 1])
        
        self.network = self._build_network()
        self.atrain = self.train()

    def _build_network(self):
        net = tf.layers.dense(self.X, 24, activation=tf.nn.relu)
        net = tf.layers.dense(net ,24, activation=tf.nn.relu)
        action_prob = tf.layers.dense(net, self.action_size, activation=tf.nn.softmax)
        return action_prob

    def train(self):
        log_lik = -self.Y*tf.log(self.network)
        log_lik_adv = log_lik * self.advantages
        loss = tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis = 1))
        train = tf.train.AdamOptimizer(0.01).minimize(loss)
        return train

    def choose_action(self, s):
        act_prob = self.sess.run(self.network, feed_dict={self.X: s})
        action = np.random.choice(self.action_size, p=act_prob[0])
        return action

def discount_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for i in reversed(range(len(r))):
        running_add = running_add * 0.99 + r[i]
        discounted_r[i] = running_add
    discounted_r = (discounted_r - discounted_r.mean())/(discounted_r.std() + 1e-7)
    return discounted_r

env = gym.make('CartPole-v0')
sess = tf.Session()
pg = PG(sess, 4, 2)
sess.run(tf.global_variables_initializer())

for i in range(200):
    xs = np.empty(shape=[0, 4])
    ys = np.empty(shape=[0, 2])
    rewards = np.empty(shape=[0, 1])
    state = env.reset()
    t = 0
    done = False
    while not done:
        x = np.reshape(state, [1, 4])
        action = pg.choose_action(x)
        xs = np.vstack([xs, x])
        y = np.zeros(2)
        y[action] = 1
        ys = np.vstack([ys, y])
        state, reward, done, _ = env.step(action)

        if done:
            reward = -1
        else:
            reward = 0
        
        rewards = np.vstack([rewards, reward])
        t += 1
        if done:
            discounted_rewards = discount_rewards(rewards)
            _ = sess.run(pg.atrain, feed_dict={pg.X: xs, pg.Y: ys, pg.advantages: discounted_rewards})

            print(i, t)