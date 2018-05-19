import tensorflow as tf

def get_copy_var_ops(*, dest_scope_name, src_scope_name):
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

class trpo:

    def __init__(self, sess, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess

        self.X = tf.placeholder(tf.float32, [None, self.state_size])

        self.pi = self._build_anet('pi')
        self.old_pi = self._build_anet('old_pi')
        self.critic = self._build_cnet()

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
        print('pi', self.sess.run(main_vars))
        print('old_pi', self.sess.run(target_vars))
        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))
        
        self.sess.run(copy_op)
        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='old_pi')
        print('pi', self.sess.run(main_vars))
        print('old_pi', self.sess.run(target_vars))
        

sess = tf.Session()
trpo = trpo(sess, 4, 2)
sess.run(tf.global_variables_initializer())
print(trpo.update_pi_network())