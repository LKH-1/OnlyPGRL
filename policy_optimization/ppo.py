import numpy as np
import tensorflow as tf
import copy

class PPOIQN:
    def __init__(self, sess):
        self.sess =sess
        self.state_size = 4
        self.action_size = 2
        self.num_support = 8
        self.gamma = 0.99
        self.batch_size = 64

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
        self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
        self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        self.act_probs, self.v_preds, self.pi_trainable = self._build_network('policy')
        self.act_probs_old, self.v_preds_old, self.old_pi_trainable = self._build_network('old_policy')

        self.assign_ops = []
        for v_old, v in zip(self.old_pi_trainable, self.pi_trainable):
            self.assign_ops.append(tf.assign(v_old, v))

        act_probs = self.act_probs * tf.one_hot(indices=self.actions, depth=self.act_probs.shape[1])
        act_probs_old = self.act_probs_old * tf.one_hot(indices=self.actions, depth=self.act_probs_old.shape[1])

        act_probs = tf.reduce_sum(act_probs, axis=1)
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.variable_scope('loss/clip'):
            act_ratios = tf.exp(tf.log(act_probs)-tf.log(act_probs_old))
            clipped_spatial_ratios = tf.clip_by_value(act_ratios, clip_value_min=1 - 0.2,
                                                      clip_value_max=1 + 0.2)
            loss_spatial_clip = tf.minimum(tf.multiply(self.gaes, act_ratios),
                                           tf.multiply(self.gaes, clipped_spatial_ratios))
            loss_spatial_clip = tf.reduce_mean(loss_spatial_clip)

        with tf.variable_scope('loss/vf'):
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, self.v_preds)
            loss_vf = tf.reduce_mean(loss_vf)

        with tf.variable_scope('loss'):
            loss = loss_spatial_clip - loss_vf
            loss = -loss

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        self.train_op = optimizer.minimize(loss, var_list=self.pi_trainable)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def train(self, obs, actions, rewards, v_preds_next, gaes):
        gaes = np.array(gaes).astype(dtype=np.float32)
        self.sess.run(self.train_op, feed_dict={self.state: obs,
                                                self.actions: actions,
                                                self.rewards: rewards,
                                                self.v_preds_next: v_preds_next,
                                                self.gaes: gaes})

    def _build_network(self, name):
        with tf.variable_scope(name):
            with tf.variable_scope('actor'):
                layer_1 = tf.layers.dense(inputs=self.state, units=64, activation=tf.nn.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.tanh)
                act_probs = tf.layers.dense(inputs=layer_2, units=self.action_size, activation=tf.nn.softmax)

            with tf.variable_scope('value'):
                layer_1 = tf.layers.dense(inputs=self.state, units=64, activation=tf.nn.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.tanh)
                v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return act_probs, v_preds, params

    def choose_action(self, obs):
        action, value = self.sess.run([self.act_probs, self.v_preds], feed_dict={self.state: [obs]})
        action = np.random.choice(self.action_size, p=action[0])
        return action, value[0][0]