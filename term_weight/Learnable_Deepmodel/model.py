# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import numpy as np
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers.python.layers import initializers




class WeightModel(object):


    def __init__(self, input_feature, local_f_len, global_f_len, input_mask, is_training=True, initializer=initializers.xavier_initializer()):
        """init all hyperparameter here"""

        # set hyperparamter
        self.local_f_len = local_f_len
        self.global_f_len = global_f_len
        self.is_training = is_training
        self.initializer = initializer
   
        # input and label
        self.input_feature = input_feature
        #self.input_mask = tf.cast(tf.not_equal(self.input_feature, -1.0), tf.float32)
        self.input_mask = tf.cast(input_mask, tf.float32)
 
        # init weights 
        self.instantiate_weights()

        # get score
        
        self.logits, self.score  = self.inference()

        # get loss
        #self.loss = self.loss()  

        # get metric 
        #self.mse = self.metric() 

        if not self.is_training:
            return
        

        # optimizer
        #self.train_op = self.train()


    def func(self, w, b, x):
        # weight square
        w_square = tf.square(w)
        
        # x subtract point b and apply relu
        x_new = tf.nn.relu(tf.subtract(x,b))

        # weighted sum
        w_x = tf.reduce_sum(tf.multiply(x_new, w_square), axis=-1)
       
          
        return w_x
         
    def inference(self):
        
        # reshape [batch, num_terms, num_features]
        shape = self.input_feature.get_shape().as_list()
        num_features = self.local_f_len + self.global_f_len
         
        features = tf.reshape(self.input_feature, [shape[0],-1,num_features]) 
        # split local and global feature
        self.input_local, self.input_global = tf.split(features, [self.local_f_len, self.global_f_len], axis=-1)
 
        # local score
        local_score = self.func(self.local_w, self.local_b, self.input_local)
 
        # global score
        global_score = self.func(self.global_w, self.global_b, self.input_global) 
         
        # local*global

        logit = tf.multiply(local_score, global_score)        
 
        # mask
        #idx_list = [idx for idx in range(0, shape[-1], num_features)]
        #mask = tf.gather(self.input_mask,idx_list, axis=1)
        adder = (1.0 - self.input_mask ) * -10000.0
        
        logit += adder
 
        # get score softmax
        score = tf.nn.softmax(logit)
         
        return logit, score

    def loss(self):
        with tf.name_scope("loss"):
            #input_mask = tf.cast(tf.not_equal(self.labels, -10.0), tf.float32)
            #labels = tf.multiply(self.labels, input_mask)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = tf.reduce_mean(loss)
            #loss = tf.losses.mean_squared_error(self.labels, self.score)
        return loss


    def metric(self):

        # mean squared error
        error = tf.square(self.labels - self.score)
        mse = tf.reduce_mean(error)        
        return mse
        
    def train(self):
        """based on the loss, use SGD to update parameter"""
        self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, tf.train.get_global_step(), self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf_contrib.layers.optimize_loss(self.loss, global_step=None,
                                                   learning_rate=self.decay_learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def train_nodecay(self):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        train_op = opt.minimize(self.loss_val)
        return train_op

    def instantiate_weights(self):
        with tf.name_scope("local_weight"):
            self.local_w = tf.get_variable("local_w", shape=[self.local_f_len],initializer=tf.initializers.random_uniform(minval=0.2, maxval=1))
            self.local_b = tf.get_variable("local_b", shape=[self.local_f_len],initializer=tf.initializers.random_uniform(minval=0.1, maxval=0.5))
        with tf.name_scope("global_weight"):
            self.global_w = tf.get_variable("global_w", shape=[self.global_f_len],initializer=tf.initializers.random_uniform(minval=0.2, maxval=1))
            self.global_b = tf.get_variable("global_b", shape=[self.global_f_len],initializer=tf.initializers.random_uniform(minval=0.1, maxval=0.5))
