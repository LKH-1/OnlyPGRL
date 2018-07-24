import tensorflow as tf
import numpy as np
import gym
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def disconut_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for i in reversed(range(len(r))):
        running_add = running_add * 0.99 + r[i]
        discounted_r[i] = running_add
    discounted_r = (discounted_r - discounted_r.mean())/(discounted_r.std() + 1e-7)
    return discounted_r

class a2c:
    def __init__(self, sess, state_size, action_size, exp_rate):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.exp_rate = exp_rate

        self.X = tf.placeholder(tf.float32, [None, self.state_size])
        self.a = tf.placeholder(tf.float32, [None, self.action_size])
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.v_ = tf.placeholder(tf.float32, [None, 1])
        self.actor, self.critic = self._bulid_net()

        self.td_error = self.r + 0.99 * self.v_ - self.critic
        self.closs = tf.square(self.td_error)

        self.log_lik = self.a * tf.log(self.actor)
        self.log_lik_adv = self.log_lik * self.td_error
        self.exp_v = tf.reduce_mean(tf.reduce_sum(self.log_lik_adv, axis=1))
        self.entropy = -tf.reduce_sum(self.actor * tf.log(self.actor))
        self.obj_func = self.exp_v + self.exp_rate * self.entropy
        self.loss = -self.obj_func

        self.total_loss = self.loss + self.closs
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        gvs = optimizer.compute_gradients(self.total_loss)
        capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)

    def learn(self, state, next_state, reward, action):
        v_ = self.sess.run(self.critic, feed_dict={self.X: next_state})
        _ = self.sess.run(self.train_op,
                          feed_dict={self.X: state, self.v_: v_, self.r: reward, self.a: action})

    def _bulid_net(self):
        net = tf.layers.dense(self.X, 24, activation=tf.tanh)
        net = tf.layers.dense(net, 24, activation=tf.tanh)

        actor = tf.layers.dense(net, 24, activation=tf.tanh)
        actor = tf.layers.dense(actor, self.action_size, activation=tf.nn.softmax)

        critic = tf.layers.dense(net, 24, activation=tf.tanh, trainable=True)
        critic = tf.layers.dense(critic, 1, activation=tf.tanh, trainable=True)

        return actor, critic

    def choose_action(self, s):
        act_prob = self.sess.run(self.actor, feed_dict={self.X: [s]})
        action = np.random.choice(self.action_size, p=act_prob[0])
        return action

sess = tf.Session()
A2C = a2c(sess, 4, 2, 0.00001)
sess.run(tf.global_variables_initializer())
env = gym.make('CartPole-v1')

spend_time = tf.placeholder(tf.float32)
rr = tf.summary.scalar('reward', spend_time)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./board/a2c_low_exp', sess.graph)

er_states = np.empty(shape=[0, 4])
er_next_states = np.empty(shape=[0, 4])
er_actions = np.empty(shape=[0, 2])
er_discounted_rewards = np.empty([0, 1])

er_size = 100000


for episodes in range(1000):
    done = False
    state = env.reset()
    rewards = np.empty(shape=[0, 1])
    t = 0
    while not done:
        t += 1
        action_pred = A2C.choose_action(state)
        next_state, reward, done, _ = env.step(action_pred)
        if done:
            reward = -1
        else:
            reward = 0

        er_states = np.vstack([er_states, state])
        er_next_states = np.vstack([er_next_states, next_state])

        rewards = np.vstack([rewards, reward])

        action = np.zeros(2)
        action[action_pred] = 1
        er_actions = np.vstack([er_actions, action])


        state = next_state

        if done:
            discounted_rewards = disconut_rewards(rewards)
            er_discounted_rewards = np.vstack([er_discounted_rewards, discounted_rewards])

            er_states = er_states[-er_size:]
            er_next_states = er_next_states[-er_size:]
            er_discounted_rewards = er_discounted_rewards[-er_size:]
            er_actions = er_actions[-er_size:]
            A2C.learn(er_states, er_next_states, er_discounted_rewards, er_actions)
            #inp = [er_states, er_next_states, er_discounted_rewards, er_actions]
            #for i in range(5):
            #    sample_indices = np.random.randint(low=0, high=er_states.shape[0],
            #                                       size=100)  # indices are in [low, high)
            #    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
            #    A2C.learn(sampled_inp[0], sampled_inp[1], sampled_inp[2], sampled_inp[3])
            summary = sess.run(merged, feed_dict={spend_time: t})
            writer.add_summary(summary, episodes)
            print(episodes, t)
