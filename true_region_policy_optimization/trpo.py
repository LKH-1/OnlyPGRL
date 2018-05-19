import tensorflow as tf
import gym
import numpy as np

def get_copy_var_ops(*, dest_scope_name, src_scope_name):
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def disconut_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for i in reversed(range(len(r))):
        running_add = running_add * 0.99 + r[i]
        discounted_r[i] = running_add
    discounted_r = (discounted_r - discounted_r.mean())/(discounted_r.std() + 1e-7)
    return discounted_r

class trpo:

    def __init__(self, sess, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess

        self.X = tf.placeholder(tf.float32, [None, self.state_size])
        self.a = tf.placeholder(tf.float32, [None, self.action_size])
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.v_ = tf.placeholder(tf.float32, [None, 1])

        self.pi = self._build_anet('pi')
        self.old_pi = self._build_anet('old_pi')
        self.critic = self._build_cnet()

        self.td_error = self.r + 0.99 * self.v_ - self.critic
        self.closs = tf.square(self.td_error)
        self.train_cop = tf.train.AdamOptimizer(0.001).minimize(self.closs)

        act_probs = self.pi * self.a
        act_probs = tf.reduce_sum(act_probs, axis=1)
        act_probs_old = self.old_pi * self.a
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - 0.2, clip_value_max=1 + 0.2)
        loss_clip = tf.minimum(tf.multiply(self.td_error, ratios), tf.multiply(self.td_error, clipped_ratios))
        loss_clip = tf.reduce_mean(loss_clip)
        entropy = -tf.reduce_sum(self.pi *
                                     tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0)), axis=1)
        entropy = tf.reduce_mean(entropy, axis=0)
        loss = loss_clip + 0.01 * entropy
        loss = -loss
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-5)
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'pi')
        self.train_aop = optimizer.minimize(loss, var_list=train_vars)

    def learn(self, state, next_state, reward, action):
        v_ = self.sess.run(self.critic, feed_dict={self.X: next_state})
        _, _ = self.sess.run([self.train_cop, self.train_aop],
                                feed_dict={self.X: state, self.v_: v_, self.r: reward, self.a: action})

    def _build_anet(self,name):
        with tf.variable_scope(name):
            anet = tf.layers.dense(self.X, 32, activation=tf.nn.relu)
            anet = tf.layers.dense(anet, 32, activation=tf.nn.relu)
            pi = tf.layers.dense(anet, self.action_size, activation=tf.nn.softmax)
        return pi

    def _build_cnet(self):
        cnet = tf.layers.dense(self.X, 32, activation=tf.nn.tanh)
        cnet = tf.layers.dense(cnet, 32, activation=tf.nn.tanh)
        critic = tf.layers.dense(cnet, 1 , activation=None)
        return critic

    def update_pi_network(self):
        copy_op = []
        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='old_pi')
        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))
        
        self.sess.run(copy_op)

    def choose_action(self, s):
        a_prob = self.sess.run(self.pi, feed_dict={self.X: [s]})
        action = np.random.choice(self.action_size, p=a_prob[0])
        return action    

sess = tf.Session()
TRPO = trpo(sess, 4, 2)
sess.run(tf.global_variables_initializer())
env = gym.make('CartPole-v0')

for episodes in range(1000):
    done = False
    state = env.reset()
    states = np.empty(shape=[0, 4])
    actions = np.empty(shape=[0, 2])
    next_states = np.empty(shape=[0, 4])
    rewards = np.empty(shape=[0, 1])
    t = 0
    while not done:
        t += 1
        action_pred = TRPO.choose_action(state)
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
            TRPO.learn(states, next_states, discounted_rewards, actions)
            TRPO.update_pi_network()
            print(episodes, t)