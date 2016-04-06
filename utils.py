import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops


# returns a matrix with [num_rows, num_cols] where the indices value is 1
def one_hot(num_cols, indices):
    num_rows = len(indices)
    mat = np.zeros((num_rows, num_cols))
    mat[np.arange(num_rows), indices] = 1
    return mat

# class batch_norm(object):
#     """Code modification of http://stackoverflow.com/a/33950177"""
#     def __init__(self, batch_size, is_train, epsilon=1e-5,
#                  momentum = 0.1, name="batch_norm", reuse=False):
#         with tf.variable_scope(name, reuse=reuse) as scope:
#             self.epsilon = epsilon
#             self.momentum = momentum
#             self.batch_size = batch_size
#             self.is_train = is_train
#             self.name = name
#             self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)

#     def __call__(self, x, reuse=False):
#         with tf.variable_scope(self.name, reuse=reuse) as scope:
#             shape = x.get_shape().as_list()
#             self.gamma = tf.get_variable("gamma", [shape[-1]],
#                                 initializer=tf.random_normal_initializer(1., 0.02))
#             self.beta = tf.get_variable("beta", [shape[-1]],
#                                 initializer=tf.constant_initializer(0.))

#             def _normalize_for_training():
#               self.mean, self.variance = tf.nn.moments(x, [0, 1, 2])
#               with tf.control_dependencies([self.mean, self.variance]):
#                 return tf.nn.batch_norm_with_global_normalization(
#                     x, self.mean, self.variance, self.beta, self.gamma,
#                     self.epsilon, scale_after_normalization=True)

#             def _normalize_for_testing():
#                 with tf.control_dependencies([self.mean, self.variance]):
#                     mean = self.ema.average(self.mean)
#                     variance = self.ema.average(self.variance)
#                     local_beta = tf.identity(self.beta)
#                     local_gamma = tf.identity(self.gamma)
#                     print 'x  = ', x==Nonesyn
#                     print 'mean = ', mean==None
#                     print 'self.mean == ', self.mean == None
#                     print 'local_beta = ', local_beta == None
#                     print 'local_gamma = ', local_gamma == None

#                     return tf.nn.batch_norm_with_global_normalization(
#                         x, mean, variance, local_beta, local_gamma,
#                         self.epsilon, scale_after_normalization=True)

#             return tf.cond(self.is_train,
#                            _normalize_for_training,
#                            _normalize_for_testing)

def batch_norm(x, is_train, epsilon=1e-5, affine=True, reuse=False, name='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        is_train:    boolean tf.Variable, true indicates training phase
        epsilon:     the epsilon used in tf.nn.batch_norm_with_global_normalization
        affine:      whether to affine-transform outputs
        name:        string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(name, reuse=reuse):
        shape = x.get_shape().as_list()
        beta = tf.Variable(tf.constant(0.0, shape=[shape[-1]]),
            name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[shape[-1]]),
            name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(is_train,
                            mean_var_with_update,
                            lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
            beta, gamma, epsilon, affine)
    return normed

def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(logits * tf.log(targets + eps) +
                              (1. - logits) * tf.log(1. - targets + eps)))

def div_round(dim, num):
    d = dim
    for n in num:
        d = int(np.ceil(d/float(n)))

    return d

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, name="conv2d", reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", with_w=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu", reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope=None, bias_start=0.0, with_w=False, reuse=False):
    with tf.variable_scope(scope or "Linear", reuse=reuse):
        shape = input_.get_shape().as_list()
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
