import numpy as np
import tensorflow as tf
import gym
import argparse
import time

from core import *


#-----------------------------------------------------------------------------------------------

class SAC():
 
    def __init__(self, sess, args):
       self.sess = sess
       self.lr = args.lr
       self.buff_size = args.buff_size
       self.gamma = args.gamma
       self.total_ep = args.total_ep 
       self.max_ep_len = args.max_ep_len
       self.batch_size = args.batch_size
       self.polyak = args.polyak  
       self.s_dim = args.s_dim
       self.a_dim = args.a_dim

       self.tf_placeholders()  
       self.networks()
       self.temperature() 
       self.losses()
       self.train_ops()


    def tf_placeholders(self):
       # tf placeholders
       self.s_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.s_dim))
       self.a_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.a_dim))
       self.s2_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.s_dim))
       self.r_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
       self.d_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

    def networks(self):
       # note: (s,a) goes as input to main network
       # but (s2,a) goes as input to target network; the a is a dummy variable here, never used! 

       # main network
       with tf.variable_scope('main'):
          self.mu, self.pi, self.logp_pi, self.q1, self.q2, self.q1_pi, self.q2_pi, self.v = mlp_actor_critic(self.s_ph, self.a_ph)
    
       # target value network (note: for target, the a_ph that goes in as input is dummy, never used!)
       with tf.variable_scope('target'):
          _, _, _, _, _, _, _, self.v_targ  = mlp_actor_critic(self.s2_ph, self.a_ph)

       # count variables
       var_counts = tuple(count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
       print(('\nNumber of parameters: \t pi: %d, \t' + 'q1: %d, \t q2: %d, \t total: %d\n')%var_counts)
       var_counts = tuple(count_vars(scope) for scope in ['target'])
       print(('\nNumber of parameters in target: \t %d\n')%var_counts)
       time.sleep(5)


    def temperature(self):
       init_value = 1.0 
       self.ent_coef = tf.get_variable('ent_coef', dtype=tf.float32, initializer=init_value) 
       self.target_entropy = -np.array(self.a_dim).astype(np.float32)


    def losses(self): 

       # min double-Q: based on TD3
       self.min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)

       # targets for Q and v
       self.q_backup = tf.stop_gradient(self.r_ph + self.gamma * (1.0-self.d_ph) * self.v_targ)

       # policy loss
       self.pi_loss = tf.reduce_mean(self.ent_coef * self.logp_pi - self.q1_pi)

       # q1 and q2 losses
       self.q1_loss = 0.5 * tf.reduce_mean((self.q1 - self.q_backup)**2)
       self.q2_loss = 0.5 * tf.reduce_mean((self.q2 - self.q_backup)**2)

       # temperature/alpha losses  
       # openai baselines code uses log_ent_coef???
       # see line 232 of https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/sac.py  
       self.ent_coef_loss = -tf.reduce_mean(self.ent_coef*tf.stop_gradient(self.logp_pi + self.target_entropy))        

       # v loss
       self.v_backup = tf.stop_gradient(self.min_q_pi - self.ent_coef * self.logp_pi)
       self.v_loss = 0.5 * tf.reduce_mean((self.v - self.v_backup) ** 2)
 

    def train_ops(self):

       # policy train op: train pi network 
       self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
       self.train_pi_op = self.pi_optimizer.minimize(self.pi_loss, var_list=get_vars('main/pi'))

       # train op for q1 
       self.q1_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
       self.train_q1_op = self.q1_optimizer.minimize(self.q1_loss, var_list=get_vars('main/q1'))

       # train op for q2
       self.q2_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
       self.train_q2_op = self.q2_optimizer.minimize(self.q2_loss, var_list=get_vars('main/q2'))

       # train op for v
       self.v_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
       self.train_v_op = self.v_optimizer.minimize(self.v_loss, var_list=get_vars('main/v'))

       # temperature/alpha
       self.entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
       self.ent_coef_op = self.entropy_optimizer.minimize(self.ent_coef_loss, var_list=get_vars('ent_coef'))

       # Polyak averaging for target variables
       self.target_update = [] 
       for theta_main, theta_targ in zip(get_vars('main'), get_vars('target')):
            self.target_update.append(tf.assign(theta_targ, self.polyak*theta_targ + (1.0-self.polyak)*theta_main))     
              


              
    def train(self, batch):
        # all ops to call during one training step
        step_ops = [self.pi_loss, self.q1_loss, self.q2_loss, self.v_loss, self.q1, self.q2, self.logp_pi, self.v, self.v_targ, self.ent_coef, self.train_pi_op, self.train_q1_op, self.train_q2_op, self.train_v_op, self.ent_coef_op, self.target_update]
        feed_dict = {self.s_ph: batch['s'], self.s2_ph: batch['s2'], self.a_ph: batch['a'], self.r_ph: batch['r'], self.d_ph: batch['done']}
        outs = self.sess.run(step_ops, feed_dict)
        return outs
 

    def init_all_vars(self):
       self.sess.run(tf.global_variables_initializer())
    
    def target_init(self):
       # initializing targets to match main variables
       target_init = []
       for theta_main, theta_targ in zip(get_vars('main'), get_vars('target')):
             target_init.append(tf.assign(theta_targ, theta_main))
       self.sess.run(target_init)


    def get_action(self, s, deterministic=False):
        if deterministic:
            act_op = self.mu
        else:
            act_op = self.pi 
 
        act = self.sess.run(act_op, feed_dict={self.s_ph: s.reshape(1,-1)})[0]

        # all act are [-1,1]
        # but we want steering: [-1,1]; acceleration: [0,1]; brake: [0,1]     
        act[1] = (act[1] + 1.0)/2.0  
        act[2] = (act[2] + 1.0)/2.0  

        # step on the gas!
        act[1] = 0.6 + 0.4*act[1]
        act[2] = 0.3*act[2]  

        return act 





