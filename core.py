import numpy as np
import tensorflow as tf
import sys


EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20

xavier = tf.contrib.layers.xavier_initializer()
bias_const = tf.constant_initializer(0.005)
rand_unif = tf.keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3)
regularizer = tf.contrib.layers.l2_regularizer(scale=1.0e-4)

#---------------------------------------------------------------------------------------


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid machine precision error, clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi

#--------------------------------------------------------------------------------------------------------------------------------

def mlp_actor_critic(s, a):
    # note: policy only uses (s)
    # q1 and q2 use (s,a)
    # q1_pi and q2_pi use only (s)

    # policy
    with tf.variable_scope('pi'):
        a_dim = a.shape.as_list()[-1]

        net = tf.layers.dense(s, units=400, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
        net = tf.layers.dense(net, units=300, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)

        mu = tf.layers.dense(net, units=a_dim, activation=None, kernel_initializer=rand_unif, bias_initializer=bias_const)
        log_std = tf.layers.dense(net, units=a_dim, activation=tf.tanh, kernel_initializer=rand_unif, bias_initializer=bias_const)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = tf.exp(log_std)

        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)

        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # q1
    with tf.variable_scope('q1'):
        x = tf.concat([s,a], axis=-1)
        net = tf.layers.dense(x, units=400, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
        net = tf.layers.dense(net, units=300, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
        q1 = tf.layers.dense(net, units=1, activation=None, kernel_initializer=rand_unif, bias_initializer=bias_const)
        q1 = tf.squeeze(q1, axis=1)

    with tf.variable_scope('q1', reuse=True):
        x = tf.concat([s,pi], axis=-1)
        net = tf.layers.dense(x, units=400, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
        net = tf.layers.dense(net, units=300, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
        q1_pi = tf.layers.dense(net, units=1, activation=None, kernel_initializer=rand_unif, bias_initializer=bias_const)
        q1_pi = tf.squeeze(q1_pi, axis=1)

    # q2
    with tf.variable_scope('q2'):
        x = tf.concat([s,a], axis=-1)
        net = tf.layers.dense(x, units=400, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
        net = tf.layers.dense(net, units=300, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
        q2 = tf.layers.dense(net, units=1, activation=None, kernel_initializer=rand_unif, bias_initializer=bias_const)
        q2 = tf.squeeze(q2, axis=1)

    with tf.variable_scope('q2', reuse=True):
        x = tf.concat([s,pi], axis=-1)
        net = tf.layers.dense(x, units=400, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
        net = tf.layers.dense(net, units=300, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
        q2_pi = tf.layers.dense(net, units=1, activation=None, kernel_initializer=rand_unif, bias_initializer=bias_const)
        q2_pi = tf.squeeze(q2_pi, axis=1)

    # v
    with tf.variable_scope('v'):
        net = tf.layers.dense(s, units=400, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
        net = tf.layers.dense(net, units=300, activation=tf.nn.relu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
        v = tf.layers.dense(net, units=1, activation=None, kernel_initializer=rand_unif, bias_initializer=bias_const)
        v = tf.squeeze(v, axis=1)

    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v

