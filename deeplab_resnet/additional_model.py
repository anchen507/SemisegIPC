# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
_BACH_NORMAL_MOMENTUM=0.95
_BACH_NORMAL_EPSILON=1e-5
DEFAULT_PADDING = 'VALID'
DEFAULT_DATAFORMAT = 'NHWC'

def GroundtruthMapMatch(inputs1,channel=21):
    with tf.variable_scope('GroundtruthMapMatch'):
        W_G1 = weight_variable([3, 3, channel, 64], name="W_G1")
        conv_G1 = tf.nn.conv2d(inputs1,W_G1,strides=[1,1,1,1],padding="SAME")
        conv_G1=tf.nn.relu(conv_G1)
        

        #conv_G1_2 = tf.nn.conv2d(inputs2,W_G1,strides=[1,1,1,1],padding="SAME")
        #conv_G1_2=tf.nn.relu(conv_G1_2)
       
        W_G2 = weight_variable([3, 3, 64, 128], name="W_G2")
        conv_G2 = tf.nn.conv2d(conv_G1,W_G2,strides=[1,1,1,1],padding="SAME")
        conv_G2=tf.nn.relu(conv_G2)
        
        #conv_G2_2 = tf.nn.conv2d(conv_G1_2,W_G2,strides=[1,1,1,1],padding="SAME")
        #conv_G2_2=tf.nn.relu(conv_G2_2)
        
        W_G3 = weight_variable([3, 3, 128, 256], name="W_G3")
        conv_G3 = tf.nn.atrous_conv2d(conv_G2,W_G3,2,padding="SAME")
        conv_G3=tf.nn.relu(conv_G3)
      
        #conv_G3_2 = tf.nn.atrous_conv2d(conv_G2_2,W_G3,2,padding="SAME")
        #conv_G3_2=tf.nn.relu(conv_G3_2)

        
        W_G4 = weight_variable([3, 3, 256, 512], name="W_G4")
        conv_G4 = tf.nn.atrous_conv2d(conv_G3,W_G4,4,padding="SAME")
        conv_G4=tf.nn.relu(conv_G4)    

        #conv_G4_2 = tf.nn.atrous_conv2d(conv_G3_2,W_G4,4,padding="SAME")
        #conv_G4_2=tf.nn.relu(conv_G4_2)


        W_G5 = weight_variable([3, 3, 512, 2], name="W_G5")
        conv_G5 = tf.nn.conv2d(conv_G4,W_G5,strides=[1,1,1,1],padding="SAME")        
        #conv_G5_2 = tf.nn.conv2d(conv_G4_2,W_G5,strides=[1,1,1,1],padding="SAME")
       
    return conv_G5,conv_G5_2



def batch_norm(inputs, name, train_phase, relu=False):
    
    output = tf.layers.batch_normalization(
        inputs,
        momentum=_BACH_NORMAL_MOMENTUM,
        epsilon=_BACH_NORMAL_EPSILON,
        training=train_phase,
        name=name+'_norm'
    )

    if relu:
        output = tf.nn.relu(output)

    return output


def zero_padding(inputs, paddings, name):
    
        pad_mat = np.array([[0,0], [paddings, paddings], [paddings, paddings], [0, 0]])
        
        return tf.pad(inputs, paddings=pad_mat, name=name+'_padding')


def conv(inputs,k_h,k_w,c_o,s_h,s_w,name,
         padding=DEFAULT_PADDING, data_format=DEFAULT_DATAFORMAT, 
         biased=True, relu=True):
    
    c_i = inputs.get_shape().as_list()[-1]
    
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    with tf.variable_scope(name) as scope:
        
        kernel = weight_variable([k_h,k_w,c_i,c_o],name='weights')
               
        output=convolve(inputs,kernel)
 
        if biased:
            
            bias = bias_variable([c_o],name='biases')
            
            output = tf.nn.bias_add(output, bias)
            
        if relu:
            
            output = tf.nn.relu(output, name=scope.name)
            
        return output
def atrous_conv(inputs,k_h,k_w,c_o,rate,name,
                padding=DEFAULT_PADDING, data_format=DEFAULT_DATAFORMAT, 
                biased=True, relu=True):
    
    c_i = inputs.get_shape().as_list()[-1]
    
    convolve = lambda i, k, r: tf.nn.atrous_conv2d(i, k, r, padding=padding)
    
    with tf.variable_scope(name) as scope:
        
        kernel = weight_variable([k_h,k_w,c_i,c_o],name='weights')
               
        output=convolve(inputs,kernel,rate)
 
        if biased:
            
            bias = bias_variable([c_o],name='biases')
            
            output = tf.nn.bias_add(output, bias)
            
        if relu:
            
            output = tf.nn.relu(output, name=scope.name)
            
        return output

def prejection_shortcut_function(inputs,name,project_stride=2):
    
    c_i=inputs.get_shape().as_list()[-1]
    
    c_o=c_i*project_stride
        
    output=conv(inputs,1,1,c_o,1,1,name+'projection_short',
                padding=DEFAULT_PADDING, data_format=DEFAULT_DATAFORMAT, 
                biased=False, relu=False)
    
    return output    

def weight_variable(shape, stddev=0.01, name=None):
    #初始化 改变initial可以改变初始化的方式
    initial=tf.truncated_normal(shape,stddev=stddev)
    if name  is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)
    
def bias_variable(shape,name=None):
    initial=tf.constant(0.0,shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name,initializer=initial)
def entropy_compute(inputs,name=None):
    
    softmax = tf.nn.softmax(inputs)
        
    epsilon = tf.constant(value=1e-10)
        
    entropy_each_category = -tf.multiply(softmax,tf.log(softmax+epsilon))
        
    return tf.reduce_sum(entropy_each_category,3,name=name)
