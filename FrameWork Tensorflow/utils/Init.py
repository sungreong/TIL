import tensorflow as tf

print(tf.__version__)
import tensorflow.contrib.slim as slim
import numpy as np
from scipy import arange

import sys
sys.path.append('/home/advice/Python/SR/Custom/')
from Activations import *
from moving_free_batch_normalization import moving_free_batch_norm
from stochastic_weight_averaging import StochasticWeightAveraging


from tensorflow.contrib.layers import *
def get_weight_variable(shape, name=None,
                        type='xavier_uniform', regularize=True, **kwargs):
    initialise_from_constant = False
    if type == 'xavier_uniform':
        initial = xavier_initializer(uniform=True, dtype=tf.float32)
    elif type == 'xavier_normal':
        initial = xavier_initializer(uniform=False, dtype=tf.float32)
    elif type == 'he_normal':
        initial = variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'he_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'caffe_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=1.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'simple':
        stddev = kwargs.get('stddev', 0.02)
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        initialise_from_constant = True
    elif type == 'bilinear':
        weights = _bilinear_upsample_weights(shape)
        initial = tf.constant(weights, shape=shape, dtype=tf.float32)
        initialise_from_constant = True
    else:
        raise ValueError('Unknown initialisation requested: %s' % type)

    if name is None:  # This keeps to option open to use unnamed Variables
        weight = tf.Variable(initial)
    else:
        if initialise_from_constant:
            weight = tf.get_variable(name, initializer=initial)
        else:
            weight = tf.get_variable(name, shape=shape, initializer=initial)
    if regularize:
        tf.add_to_collection('weight_variables', weight)
    return weight 



def focal_loss_sigmoid(labels,logits,alpha=0.25 , gamma=2):
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(  tf.maximum(y_pred , 1e-14 )   )-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log( tf.maximum( 1-y_pred ,  1e-14 ) ) 
    return L


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)


def spectral_norm(w, iteration=1 , name = None):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable(name , [1, w_shape[-1]], 
                        initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm

def log(x):
    return tf.log( tf.maximum( x , 1e-10) )

def linear(input, output_dim, scope=None, stddev=1.0):
    ## https://github.com/AYLIEN/gan-intro/blob/master/gan.py
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b

def layer(prev = None , shape1 = None , shape2 = None , 
          name = None , activation = tf.nn.leaky_relu , usebias = True , 
          final = False , SN = True , Type = None , phase = None ) :
    with tf.variable_scope(name):
        select_w_init = np.random.randint(0, 2, size=1 )[0]
        seed_n = np.random.randint(1, 1000, size=1 )[0]
        relu_w_init = [tf.keras.initializers.he_uniform(seed = seed_n) ,
                       tf.keras.initializers.he_normal(seed = seed_n)][select_w_init]
        tanh_w_init = [tf.keras.initializers.glorot_normal(seed = seed_n) ,
                       tf.keras.initializers.glorot_uniform(seed = seed_n)][select_w_init]
        s_elu_w_init = [tf.keras.initializers.lecun_normal(seed = seed_n) ,
                       tf.keras.initializers.lecun_uniform(seed = seed_n)][select_w_init]
        nomal_w_init = tf.keras.initializers.truncated_normal(seed = seed_n)
        if activation in [tf.nn.leaky_relu, tf.nn.relu] :  init = relu_w_init
        elif activation in [tf.nn.tanh , tf.nn.softmax] :  init = tanh_w_init
        elif activation in [tf.nn.selu , tf.nn.elu , mish] :      init = s_elu_w_init
        else : 
            if final : init = tanh_w_init
            else : init = nomal_w_init
        ###
        if usebias :b1 = tf.get_variable("Bias_" + str(name) , 
                                         shape = [shape2] , dtype = tf.float32 , 
                                         initializer = tf.constant_initializer(0.0))
        else :b1 = tf.constatnt(0.0 , shape = [shape2] , dtype = tf.float32 )

        W1 = tf.get_variable("Weight_" + str(name) , dtype = tf.float32 , 
                             shape = [shape1 , shape2] , initializer = init)
        
        
        if final == True :
            if SN == True :
                W2 = spectral_norm(W1 , name = "SN" + str(name))
                layer = tf.matmul( prev , W2) + b1
            else :
                layer = tf.matmul( prev , W1 ) + b1
        else :
            if SN == True :
                W2 = spectral_norm(W1 , name = "SN" + str(name))
                layer = tf.matmul( prev , W2) + b1
            else : layer = tf.matmul( prev , W1) + b1
            ################################
            if Type == "SWA" :
                layer = moving_free_batch_norm(layer, axis=-1, 
                                               training=is_training_bn,
                                               use_moving_statistics=use_moving_statistics, 
                                               momentum=0.99)
            elif Type == "Self_Normal" :
                if activation == nalu :layer = activation(layer ,2 , name = "NALU_" + name )
                else :layer = activation(layer)
                layer = tf.contrib.nn.alpha_dropout(layer , 0.8)
            elif Type == "Batch_Normalization" :
                layer = tf.contrib.layers.batch_norm(layer, 
                                                     center=True, scale=True, 
                                                     is_training=True, # phase
                                                     scope='bn')
            elif Type == "Instance_Normalization" :layer = tf.contrib.layers.instance_norm(layer)
            else : pass
            if Type == "Self_Normal" : pass
            else : 
                if activation == nalu :
                    layer = activation(layer ,2 , name = "NALU_" + name )
                else : layer = activation(layer)
        return layer



def weights_init(shape):
    '''
    Weights initialization helper function.
    
    Input(s): shape - Type: int list, Example: [5, 5, 32, 32], This parameter is used to define dimensions of weights tensor
    
    Output: tensor of weights in shape defined with the input to this function
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def bias_init(shape, bias_init=0.05):
    '''
    Bias initialization helper function.
    
    Input(s): shape - Type: int list, Example: [32], This parameter is used to define dimensions of bias tensor.
              bias_value - Type: float number, Example: 0.01, This parameter is set to be value of bias tensor.
    
    Output: tensor of biases in shape defined with the input to this function
    '''
    return tf.Variable(tf.constant(bias_init, shape=shape))

def highway_fc_layer(input, shape1 = None , shape2 = None , carry_b = -2.0, activation=tf.nn.relu):
    '''
    The function used to crate Highway fully connected layer in the network.
    
    Inputs: input - data input
            hidden_layer_size - number of neurons in the hidden layers (highway layers)
            carry_b -  value for the carry bias used in transform gate
            activation - non-linear function used at this layer
    '''
    #Step 1. Define weights and biases for the activation gate
    weights_normal = weights_init([shape1, shape2])
    bias_normal = bias_init([shape2])
    #Step 2. Define weights and biases for the transform gate
    weights_transform = weights_init([shape1, shape2])
    bias_transform = bias_init(shape=[shape2], bias_init=carry_b)
    ## extra
    input_transform = weights_init([shape1, shape2])
    #Step 3. calculate activation gate
    H = activation(tf.matmul(input, weights_normal) + bias_normal, name="Input_gate")
    #Step 4. calculate transform game
    T = tf.nn.sigmoid(tf.matmul(input, weights_transform) +bias_transform, name="T_gate")
    #Step 5. calculate carry get (1 - T)
    C = tf.subtract(1.0, T, name='C_gate')
    # y = (H * T) + (x * C)
    #Final step 6. campute the output from the highway fully connected layer
    y = tf.add(tf.multiply(H, T), tf.multiply(tf.matmul(input,input_transform) , C), name='output_highway')
    return y
