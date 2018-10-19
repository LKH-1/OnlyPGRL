import tensorflow as tf
import copy


class PPOTrain:
    def __init__(self, sess, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.005):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """
        self.sess = sess
        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.space = tf.placeholder(dtype=tf.int32, shape=[None], name='space')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs
        
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)
        
        action_probs = tf.clip_by_value(act_probs, 1e-10, 1.0)

        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        action_probs_old = tf.clip_by_value(act_probs_old, 1e-10, 1.0)


        with tf.variable_scope('loss/clip'):
            spatial_ratios = tf.exp(tf.log(action_probs)-tf.log(action_probs_old))
            clipped_spatial_ratios = tf.clip_by_value(spatial_ratios, clip_value_min=1-clip_value, clip_value_max=1+clip_value)
            loss_spatial_clip = tf.minimum(tf.multiply(self.gaes, spatial_ratios), tf.multiply(self.gaes, clipped_spatial_ratios))
            loss_spatial_clip = tf.reduce_mean(loss_spatial_clip)
            tf.summary.scalar('loss_spatial', loss_spatial_clip)
        
        with tf.variable_scope('loss/vf'):
            v_preds = self.Policy.v_preds
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            tf.summary.scalar('loss_vf', loss_vf)

        with tf.variable_scope('loss/entropy'):
            act_probs = self.Policy.act_probs

            act_entropy = -tf.reduce_sum(self.Policy.act_probs * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            act_entropy = tf.reduce_mean(act_entropy, axis=0)

            entropy = act_entropy
            tf.summary.scalar('entropy', entropy)

        with tf.variable_scope('loss'):
            loss = loss_spatial_clip - c_1 * loss_vf + c_2 * entropy
            loss = -loss  # minimize -loss == maximize loss
            tf.summary.scalar('loss', loss)

        self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)
        
    def train(self, obs, actions, rewards, v_preds_next, gaes):
        self.sess.run([self.train_op], feed_dict={self.Policy.obs: obs,
                                                                 self.Old_Policy.obs: obs,
                                                                 self.actions: actions,
                                                                 self.rewards: rewards,
                                                                 self.v_preds_next: v_preds_next,
                                                                 self.gaes: gaes})

    def get_summary(self, obs, actions, rewards, v_preds_next, gaes):
        return self.sess.run([self.merged], feed_dict={self.Policy.obs: obs,
                                                                      self.Old_Policy.obs: obs,
                                                                      self.actions: actions,
                                                                      self.rewards: rewards,
                                                                      self.v_preds_next: v_preds_next,
                                                                      self.gaes: gaes})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return self.sess.run(self.assign_ops)
    
    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes