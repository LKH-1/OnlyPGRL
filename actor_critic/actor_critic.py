import tensorflow as tf
import numpy as np
import gym

def disconut_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for i in reversed(range(len(r))):
        running_add = running_add * 0.99 + r[i]
        discounted_r[i] = running_add
    discounted_r = (discounted_r - discounted_r.mean())/(discounted_r.std() + 1e-7)
    return discounted_r

class a2c:
    def __init__(self, sess, state_size, action_size):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size

        self.X = tf.placeholder(tf.float32, [None, self.state_size])
        self.a = tf.placeholder(tf.float32, [None, self.action_size])
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.v_ = tf.placeholder(tf.float32, [None, 1])
        self.actor, self.critic = self._bulid_net()

        self.td_error = self.r + 0.99 * self.v_ - self.critic
        self.closs = tf.square(self.td_error)
        self.train_cop = tf.train.AdamOptimizer(0.001).minimize(self.closs)

        self.log_lik = -self.a * tf.log(self.actor)
        self.log_lik_adv = self.log_lik * self.td_error
        self.exp_v = tf.reduce_mean(tf.reduce_sum(self.log_lik_adv, axis=1))
        self.train_aop = tf.train.AdamOptimizer(0.001).minimize(self.exp_v)

    def learn(self, state, next_state, reward, action):
        v_ = self.sess.run(self.critic, feed_dict={self.X: next_state})
        _ = self.sess.run(self.train_cop,
                          feed_dict={self.X: state, self.v_: v_, self.r: reward})
        _ = self.sess.run(self.train_aop,
                            feed_dict={self.X: state, self.a: action, self.r: reward})

    def _bulid_net(self):
        net = tf.layers.dense(self.X, 24, activation=tf.nn.relu)
        net = tf.layers.dense(net, 24, activation=tf.nn.relu)

        actor = tf.layers.dense(net, 24, activation=tf.nn.relu)
        actor = tf.layers.dense(actor, self.action_size, activation=tf.nn.softmax)

        critic = tf.layers.dense(net, 24, activation=tf.nn.tanh)
        critic = tf.layers.dense(critic, 1, activation=None)

        return actor, critic

    def choose_action(self, s):
        act_prob = self.sess.run(self.actor, feed_dict={self.X: [s]})
        action = np.random.choice(self.action_size, p=act_prob[0])
        return action

sess = tf.Session()
A2C = a2c(sess, 4, 2)
sess.run(tf.global_variables_initializer())
env = gym.make('CartPole-v0')

for episodes in range(100):
    done = False
    state = env.reset()
    states = np.empty(shape=[0, 4])
    actions = np.empty(shape=[0, 2])
    next_states = np.empty(shape=[0, 4])
    rewards = np.empty(shape=[0, 1])
    t = 0
    while not done:
        t += 1
        action_pred = A2C.choose_action(state)
        print(action_pred)
        next_state, reward, done, _ = env.step(action_pred)
        if done:
            reward = -1
        else:
            reward = 0
        states = np.vstack([states, state])
        next_states = np.vstack([next_states, next_state])
        rewards = np.vstack([rewards, reward])
        action = np.zeros(2)
        action[action_pred] = 1
        actions = np.vstack([actions, action])
        state = next_state

        if done:
            discounted_rewards = disconut_rewards(rewards)
            A2C.learn(states, next_states, discounted_rewards, actions)
            print(episodes, t)