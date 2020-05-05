import tensorflow as tf
import numpy as np
from scipy import arange


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