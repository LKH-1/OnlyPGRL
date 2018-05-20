import tensorflow as tf

class Policy_network:
    def __init__(self, sess, name, state_size, action_size, temp=0.1):
        
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, [None, self.state_size], name='obs')

            with tf.variable_scope('policy_net'):
                anet = tf.layers.dense(self.x, 32, activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
                anet = tf.layers.dense(anet, 32, activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
                anet = tf.layers.dense(anet, self.action_size, activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
                self.act_probs = tf.layers.dense(tf.divide(anet, 0.1), self.action_size, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                cnet = tf.layers.dense(self.x, 32, activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
                cnet = tf.layers.dense(cnet, 32, activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
                cnet = tf.layers.dense(cnet, 32, activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
                self.v_preds = tf.layers.dense(cnet, 1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name
    
    def act(self, s, stochastic=True):
        if stochastic:
            return self.sess.run([self.act_stochastic, self.v_preds], feed_dict={self.x: s})
        else:
            return self.sess.run([self.act_deterministic, self.v_preds], feed_dict={self.x : s})
    
    def get_action_prob(self, s):
        return self.sess.run(self.act_probs, feed_dict={self.x: s})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)