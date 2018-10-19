import numpy as np
import tensorflow as tf


class Policy_net:
    def __init__(self, sess, name: str, temp=0.1):

        self.sess = sess
        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='obs')
            with tf.variable_scope('policy_net'):
                dense_1 = tf.layers.dense(inputs=self.obs, units=256, activation=tf.nn.relu)
                self.act_probs = tf.layers.dense(inputs=dense_1, units=2, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                dense_2 = tf.layers.dense(inputs=dense_1, units=64, activation=tf.nn.relu)
                self.v_preds = tf.layers.dense(inputs=dense_2, units=1, activation=None, trainable=True, kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.scope = tf.get_variable_scope().name

    def act(self, obs):
        action, value = self.sess.run([self.act_probs, self.v_preds], feed_dict={self.obs: [obs]})
        action = np.random.choice(2, p=action[0])
        return action, value[0][0]

    def get_action_prob(self, obs):
        return self.sess.run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)