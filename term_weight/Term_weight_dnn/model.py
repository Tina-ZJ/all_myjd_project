# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import numpy as np
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers.python.layers import initializers




class Semantic:
    def __init__(self, input_query, pos_item, neg_item, margin, keep_prob, learning_rate, decay_steps, decay_rate, batch_size, vocab_size, embed_size, hidden_size, is_training, initializer=initializers.xavier_initializer(), clip_gradients=5.0):
        """init all hyperparameter here"""
        # set hyperparamter
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.95)
        self.initializer = initializer
        self.clip_gradients=clip_gradients
        self.margin = margin
   
        # input and mask    
        self.input_query = input_query
        self.query_mask = tf.cast(tf.not_equal(self.input_query, 0), tf.float32)
        
        self.pos_item = pos_item
        self.pos_mask = tf.cast(tf.not_equal(self.pos_item, 0), tf.float32)
        
        self.neg_item = neg_item
        self.neg_mask = tf.cast(tf.not_equal(self.neg_item, 0), tf.float32)

    


        self.instantiate_dnn()
        self.loss = self.inference_dnn()   
        self.pos_acc, self.neg_acc = self.metric() 

        if not self.is_training:
            return
        

        # optimizer
        self.train_op = self.train()


    def share_dnn(self, input_embs):
        
        with tf.variable_scope("dnn", reuse=tf.AUTO_REUSE):
            h0 = tf.layers.dense(input_embs, 768, activation=tf.nn.relu)
            h1 = tf.layers.dense(h0, 512, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, self.hidden_size, activation=tf.nn.relu)
        return h2
             

    def attention(self, input_embs, input_mask):
        dot = tf.multiply(input_embs, self.context)
        attention = tf.reduce_sum(dot, axis=-1)
        # mask
        attention+=(1.0 - input_mask)*-10000.0
        attention = tf.nn.softmax(attention) 
        return attention
 
    def inference_dnn(self):
        # get emb
        self.query_embs = tf.nn.embedding_lookup(self.Embedding, self.input_query)
        self.pos_embs = tf.nn.embedding_lookup(self.Embedding, self.pos_item)
        self.neg_embs = tf.nn.embedding_lookup(self.Embedding, self.neg_item)

        # dnn
        self.query_output = self.share_dnn(self.query_embs)  
        self.pos_output = self.share_dnn(self.pos_embs)
        self.neg_output = self.share_dnn(self.neg_embs)

        # get weight 
        self.query_weight = self.attention(self.query_output, self.query_mask)
        self.pos_weight = self.attention(self.pos_output, self.pos_mask) 
        self.neg_weight = self.attention(self.neg_output, self.neg_mask) 
        
        # get final embeddings
        
        query_emb_f = self.query_output * tf.expand_dims(self.query_weight, -1)
        query_emb_f = tf.reduce_sum(query_emb_f, 1) 
        pos_emb_f = self.pos_output * tf.expand_dims(self.pos_weight, -1) 
        pos_emb_f = tf.reduce_sum(pos_emb_f, 1) 
        neg_emb_f = self.neg_output * tf.expand_dims(self.neg_weight, -1) 
        neg_emb_f = tf.reduce_sum(neg_emb_f, 1) 
        

        # normalize
        self.query_emb_f = tf.nn.l2_normalize(query_emb_f)
        self.pos_emb_f = tf.nn.l2_normalize(pos_emb_f)
        self.neg_emb_f = tf.nn.l2_normalize(neg_emb_f)
 
        # comput query, item cos similar

        dot = tf.multiply(self.query_emb_f, self.pos_emb_f) 
        self.pos_similar = tf.reduce_sum(dot, axis=1)
        dot = tf.multiply(self.query_emb_f, self.neg_emb_f) 
        self.neg_similar = tf.reduce_sum(dot, axis=1)
        
        # triple loss
        loss = self.neg_similar - self.pos_similar + self.margin 
        #loss = tf.reduce_mean(loss)
        loss = tf.maximum(loss, tf.zeros_like(loss))
        loss = tf.reduce_sum(loss)
        return loss

    def metric(self):
        pos_predict = tf.cast(tf.greater(self.pos_similar, 0.5), tf.float32)
        pos_acc = tf.reduce_mean(pos_predict)
        neg_predict = tf.cast(tf.less(self.neg_similar, 0.5), tf.float32)
        neg_acc = tf.reduce_mean(neg_predict)
        return pos_acc, neg_acc
        
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

    def instantiate_dnn(self):
        with tf.name_scope("embedding_projection"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)
        with tf.name_scope("attention"):
            self.context = tf.get_variable("context", shape=[self.hidden_size], initializer=self.initializer)
