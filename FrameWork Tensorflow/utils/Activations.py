import tensorflow as tf
import numpy as np
from scipy import arange

"""
Reference
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
"""
def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

def brelu(x):
    """Bipolar ReLU as in https://arxiv.org/abs/1709.04054."""
    x_shape = shape_list(x)
    #x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 2]), 2, axis=-1)
    x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 1]), 2, axis=0)
    y1 = tf.nn.relu(x1)
    y2 = -tf.nn.relu(-x2)
    return tf.reshape(tf.concat([y1, y2], axis=-1), x_shape)


def belu(x):
    """Bipolar ELU as in https://arxiv.org/abs/1709.04054."""
    x_shape = shape_list(x)
    #x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 2]), 2, axis=-1)
    x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 1]), 2, axis=0)
    y1 = tf.nn.elu(x1)
    y2 = -tf.nn.elu(-x2)
    return tf.reshape(tf.concat([y1, y2], axis=-1), x_shape)


def bent_identity(x) :
    return tf.div(tf.sqrt(tf.square(x) + 1) -1 ,2) + x
    
    
def tf_mish(x) :
    return x * tf.nn.tanh(tf.nn.softplus(x))

def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
    x: float Tensor to perform activation.
    Returns:
    x with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.nn.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def mish(x) :
    return x * tf.nn.tanh( tf.nn.softplus(x))

def gauss(x) :
    """ 0 ~ 1 로 하는 가우시안 함수 """
    return tf.exp(-x**2)

def soft_cliping(alpha = 0.5 , x=None ) :
    """ 0 ~ 1 로 하는 새로운 함수 alpha라는 하이퍼 파라미터가 존재함"""
    first = tf.div(1.0 , alpha)
    second = tf.log( tf.div(tf.add(1.0, tf.exp( tf.multiply(alpha , x) )) , 
                            tf.add(1.0 , tf.exp( tf.multiply(alpha, (tf.add(x , -1.0)) )))))
    return tf.multiply(first , second )

## https://cup-of-char.com/writing-activation-functions-from-mostly-scratch-in-python/

def tf_sqnlsig(x):   #tensorflow SQNLsigmoid
    """https://cup-of-char.com/writing-activation-functions-from-mostly-scratch-in-python/"""
    u=tf.clip_by_value(x,-2,2)
    a = u
    b= tf.negative(tf.abs(u))
    wsq = (tf.multiply(a,b))/4.0
    y = tf.add(tf.multiply(tf.add(u,wsq),0.5),0.5)
    return y

#     alphas = tf.get_variable('alpha', _x.get_shape()[-1],
#                        initializer=tf.constant_initializer(0.0),
#                         dtype=tf.float32)
def parametric_relu(_x):
    alphas = tf.Variable(tf.zeros(_x.get_shape()[-1]),
                         name = "prelu" ,
                         dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def tf_sqnl(x): #tensorflow SQNL
    """https://cup-of-char.com/writing-activation-functions-from-mostly-scratch-in-python/"""
    #tf.cond(x>2,lambda: tf.multiply(2,1),lambda:tf.multiply(x,1))
    #tf.cond(tf.less(x,-2),lambda: -2,lambda:tf.multiply(x,1))
    u=tf.clip_by_value(x,-2,2)
    a = u
    b= tf.negative(tf.abs(u))
    wsq = (tf.multiply(a,b))/4.0
    y = tf.add(u,wsq)
    return y

# 출처: https://creamyforest.tistory.com/48 [Dohyun's Blog]
def Relu2(x):
    return tf.minimum(tf.maximum(x ,0.0) , 2.0)

def tf_nalu(input_layer, num_outputs , epsilon=1e-30 ):
    """ Neural Arithmetic Logic Unit tesnorflow layer
    Arguments:
    input_layer - A Tensor representing previous layer
    num_outputs - number of ouput units 
    Returns:
    A tensor representing the output of NALU
    https://github.com/grananqvist/NALU-tf/blob/master/nalu.py
    가장 쉽게 되는 듯??
    """
    shape = (int(input_layer.shape[-1]), num_outputs)

    # define variables
    W_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    M_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    G = tf.Variable(tf.truncated_normal(shape, stddev=0.02))

    # operations according to paper
    W = tf.tanh(W_hat) * tf.sigmoid(M_hat)
    m = tf.exp(tf.matmul(tf.log(tf.abs(input_layer) + epsilon ), W))
    g = tf.sigmoid(tf.matmul(input_layer, G))
    a = tf.matmul(input_layer, W)
    out = g * a + (1 - g) * m

    return out


def nac(x, num_outputs , name=None, reuse=None):
    """
    NAC as in https://arxiv.org/abs/1808.00508.    
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
    """
    with tf.variable_scope(name, default_name="nac", values=[x], reuse=reuse):
        x_shape = shape_list(x)
        w = tf.get_variable("w", [x_shape[-1], num_outputs])
        m = tf.get_variable("m", [x_shape[-1], num_outputs])
        w = tf.tanh(w) * tf.nn.sigmoid(m)
        x_flat = tf.reshape(x, [-1, x_shape[-1]])
        res_flat = tf.matmul(x_flat, w)
        return tf.reshape(res_flat, x_shape[:-1] + [num_outputs])

def nalu(x, num_outputs, epsilon=1e-30, name=None, reuse=None):
    """
    NALU as in https://arxiv.org/abs/1808.00508.
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
    """
    with tf.variable_scope(name, default_name="nalu", values=[x], reuse=reuse):
        x_shape = shape_list(x)
        x_flat = tf.reshape(x, [-1, x_shape[-1]])
        gw = tf.get_variable("w", [x_shape[-1], num_outputs])
        g = tf.nn.sigmoid(tf.matmul(x_flat, gw))
        g = tf.reshape(g, x_shape[:-1] + [num_outputs])
        a = nac(x, num_outputs, name="nac_lin")
        log_x = tf.log(tf.abs(x) + epsilon)
        m = nac(log_x, num_outputs, name="nac_log")
        return g * a + (1 - g) * tf.exp(m)
