# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import numpy as np
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn

from nn_impl_v2 import nce_loss



class HierarchicalAttention:
    def __init__(self, keep_prob, cid2_weight, cid3_weight, brand_weight, relation_matrix, product_matrix, brand_matrix, input_x, input_y_first, input_y_second, input_product, input_brand, input_tag, fc_size, attention_unit_size, alpha, threshold, num_classes_first, num_classes_second, num_classes_third, num_classes_product, num_classes_brand, num_tags, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size, hidden_size, is_training, initializer=initializers.xavier_initializer(), clip_gradients=5.0):
        """init all hyperparameter here"""
        # set hyperparamter
        self.fc_size = fc_size
        self.attention_unit_size = attention_unit_size
        self.alpha = alpha
        self.threshold = threshold
        self.num_classes_first = num_classes_first
        self.num_classes_second = num_classes_second
        self.num_classes_third = num_classes_third
        self.num_classes_product = num_classes_product
        self.num_classes_brand = num_classes_brand
        self.num_tags = num_tags
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.95)
        self.initializer = initializer
        self.clip_gradients=clip_gradients

        # cid2 to cid3 relation matrix, cid3 to product and cid3 to brand matrix
        self.relation_matrix = tf.Variable(relation_matrix, trainable=False, name="relation_matrix")
        self.product_matrix = tf.Variable(product_matrix, trainable=False, name="product_matrix")
        self.brand_matrix = tf.Variable(brand_matrix, trainable=False, name="brand_matrix")


        # cid2, cid3, brand weight
        self.cid2_loss_weight = tf.constant(value=cid2_weight, dtype=tf.float32)
        self.cid3_loss_weight = tf.constant(value=cid3_weight, dtype=tf.float32)
        self.brand_loss_weight = tf.constant(value=brand_weight, dtype=tf.float32)
       
         #self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_x = input_x
        # sequence true length
        self.length = tf.reduce_sum(tf.sign(self.input_x), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)

        self.input_mask = tf.cast(tf.not_equal(self.input_x, 0), tf.float32)

        #### all third level    ###########
        #self.input_y_first = tf.placeholder(tf.float32, [None, self.num_classes_first],name="input_y_first")
        #self.input_y_second = tf.placeholder(tf.float32, [None, self.num_classes_second],name="input_y_second")
        #self.input_y_third = tf.placeholder(tf.float32, [None, self.num_classes_third],name="input_y_third")
        #self.input_tag = tf.placeholder(tf.int32, [None, None], name="input_tag")
        self.input_y_first = input_y_first
        self.input_y_second = input_y_second
        self.input_product = input_product
        self.input_brand = input_brand
        self.input_tag = input_tag
    
        
        #self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = keep_prob

        #self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        #self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        #self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        # lstm output
        self.gru_out, self.gru_out_pool = self.BiGRU()

        # for tag head
        self.tag_logits = self.Ner(self.gru_out)
        tag_predictions = tf.nn.softmax(self.tag_logits, axis=-1) 
        self.tag_predictions = tf.argmax(tag_predictions, axis=-1) 
        self.tag_predictions = tf.identity(self.tag_predictions, name='tag_predicitions')
 
        # first level
        self.first_att_weight, self.first_att_out = self.attention(self.gru_out, self.num_classes_first, name="first-")
        self.first_local_input = tf.concat([self.gru_out_pool, self.first_att_out], axis=1)
        #self.first_local_input = self.first_att_out
        self.first_local_fc_out = self.fc_layer(self.first_local_input, name="first-local-")
        self.first_logits, self.first_scores, self.first_visual = self.local_layer(
            self.first_local_fc_out, self.first_att_weight, self.num_classes_first, name="first-")

        # second level
        self.second_att_input = tf.multiply(self.gru_out, tf.expand_dims(self.first_visual, -1))
        self.second_local_input = tf.reduce_sum(self.second_att_input, axis=1)
        #self.second_att_weight, self.second_att_out = self.attention(self.second_att_input, self.num_classes_second, name="second-")
        #self.second_local_input = tf.concat([self.gru_out_pool, self.second_att_out], axis=1)
        #self.second_local_fc_out = self.fc_layer(self.second_local_input, name="second-local-")
        #self.second_logits, self.second_scores, self.second_visual = self.local_layer(
        #    self.second_local_fc_out, self.second_att_weight, self.num_classes_second, name="second-")

        


        # global
        #self.ham_out = tf.concat([self.first_local_fc_out, self.second_local_fc_out, self.third_local_fc_out], axis=1)

        # Fully layer
        #self.fc_out = self.fc_layer(self.ham_out)
        #self.fc_out = self.ham_out
        
        # add by another
        #self.second_local_output = tf.concat([self.gru_out_pool, self.second_local_input], axis=1)
        self.fc_out = self.fc_layer(self.second_local_input)
       
        # dropout 
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.fc_out, self.dropout_keep_prob)
        # global output
        self.second_logits, self.second_scores = self.global_layer(self.num_classes_second, 'cid_class')
        #self.product_logits, self.product_scores = self.global_layer(self.num_classes_product, 'product_class')
        #self.brand_logits, self.brand_scores = self.global_layer(self.num_classes_brand, 'brand_class')
        self.brand_w, self.brand_b, self.brand_logits, self.brand_scores = self.get_wb(self.num_classes_brand, self.is_training, 'brand_class')
        self.product_w, self.product_b, self.product_logits, self.product_scores = self.get_wb(self.num_classes_product, self.is_training, 'product_class')
         
        self.predictions = tf.reshape(self.second_scores,[-1, self.num_classes_second], name='prediction')
        self.product_predictions = tf.reshape(self.product_scores,[-1, self.num_classes_product], name='product_prediction')
        self.brand_predictions = tf.reshape(self.brand_scores,[-1, self.num_classes_brand], name='brand_prediction')
       
        self.cid_precision, self.cid_recall, self.cid_f1 = self.metric(self.predictions, self.input_y_second, 'cid_') 
        #self.product_precision, self.product_recall, self.product_f1 = self.metric(self.product_predictions, self.input_product, 'product_') 
        #self.brand_precision, self.brand_recall, self.brand_f1 = self.metric(self.brand_predictions, self.input_brand, 'brand_') 
        
        # final output combine local and global output
        #with tf.name_scope("output"):
        #    self.local_scores = tf.concat([self.first_scores, self.second_scores, self.third_scores], axis=1)
        #    self.scores = tf.add(self.alpha * self.global_scores, (1 - self.alpha) * self.local_scores, name="scores")
        

        # for tag acc
        correct = tf.cast(tf.equal(tf.cast(self.tag_predictions, tf.int64) - self.input_tag, 0), tf.float32)
        correct = tf.reduce_sum(self.input_mask * correct) 
        self.tag_acc = tf.div(correct, tf.cast(tf.reduce_sum(self.length), tf.float32))
            
       

        if not self.is_training:
            return
        
        # multi task loss 
        self.loss_val = self.multi_task_loss()

        # optimizer
        self.train_op = self.train()
      

    def metric(self, predictions, labels, name):
        tp = tf.reduce_sum(tf.cast(tf.greater(predictions, self.threshold), tf.float32) * tf.cast(tf.equal(labels, 1), tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.less(predictions, self.threshold), tf.float32) * tf.cast(tf.equal(labels, 0), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.greater(predictions, self.threshold), tf.float32) * tf.cast(tf.equal(labels, 0), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.less(predictions, self.threshold), tf.float32) * tf.cast(tf.equal(labels, 1), tf.float32))
        precision = tf.div(tp, tp+fp, name=name+'precision')
        recall = tf.div(tp, tp+fn, name=name+'recall')
        f1 = tf.div(2*precision*recall, precision+recall, name=name+'f1')
        return precision, recall, f1
 
    def global_layer(self,  num_class,  name):
        #dropout
        with tf.name_scope(name):
            num_units = self.h_drop.get_shape().as_list()[-1]
            W = tf.Variable(tf.truncated_normal(shape=[num_units, num_class],
                                                stddev=0.1, dtype=tf.float32), name="W_"+name)
            b = tf.Variable(tf.constant(value=0.1, shape=[num_class],dtype=tf.float32), name="b_"+name)
            global_logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            global_scores = tf.sigmoid(global_logits, name="scores")
        return global_logits, global_scores

    def get_wb(self, num_class, flag, name):
        global_logits, global_scores = [], []
        with tf.name_scope(name):
            num_units = self.h_drop.get_shape().as_list()[-1]
            W = tf.Variable(tf.truncated_normal(shape=[num_class, num_units],
                                                stddev=0.1, dtype=tf.float32), name="W_"+name)
            b = tf.Variable(tf.constant(value=0.1, shape=[num_class],dtype=tf.float32), name="b_"+name)
            if not flag:
                global_logits = tf.nn.xw_plus_b(self.h_drop, tf.transpose(W), b, name="logits")
                global_scores = tf.sigmoid(global_logits, name="scores")
        return W, b, global_logits, global_scores
             
         
    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)
        lstm_fw_cell = rnn.GRUCell(self.hidden_size)
        lstm_bw_cell = rnn.GRUCell(self.hidden_size)
        ## dropout to rnn
        #if self.is_training:
        #    lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
        #    lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
        outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, sequence_length=self.length, dtype=tf.float32)
        self.hidden_state = tf.concat(outputs, axis=2)
        ## attention ##
        hidden_state_ = tf.reshape(self.hidden_state, shape=[-1, self.hidden_size *2])
        u = tf.nn.tanh(tf.matmul(hidden_state_,self.W_w_attention_word,)+self.W_b_attention_word)
        u = tf.reshape(u, shape=[self.batch_size, -1, self.hidden_size * 2])
        uv = tf.multiply(u, self.context_vecotor_word)
        uv = tf.reduce_sum(uv, axis=2)
        ## Mask ##
        #uv = tf.multiply(self.input_mask, uv)
        #p_attention = tf.nn.softmax(uv - tf.reduce_max(uv,-1,keepdims=True))
        ## Mask 2 ways##
        uv+=(1.0 - tf.cast(self.input_mask, tf.float32))*-10000.0
        self.attention = tf.nn.softmax(uv-tf.reduce_max(uv,-1,keepdims=True))

        ## Mask attention and to sum(1)##
        #p_attention = tf.multiply(self.input_mask, p_attention)
        #p_attention_sum = tf.reduce_sum(p_attention, axis=1)
        #p_attention_sum = tf.expand_dims(p_attention_sum, -1)
        #self.attention = tf.divide(p_attention, p_attention_sum)
        #self.attention = p_attention
        
        ## output ##
        hidden_new = self.hidden_state*tf.expand_dims(self.attention,-1) 
        sentence_representation = tf.reduce_sum(hidden_new, 1) 

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(sentence_representation,keep_prob=self.dropout_keep_prob)

        # 5. logits
        with tf.name_scope("output"):
                logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection

        #tag head logits
        if self.tag_flag:
            with tf.name_scope("tag_output"):
                hidden_state_tag = tf.reshape(hidden_new, shape=[-1,self.hidden_size*2])
                self.tag_drop = tf.nn.dropout(hidden_state_tag, keep_prob=self.dropout_keep_prob)
                self.tag_logitss = tf.matmul(self.tag_drop, self.W_tag) + self.b_tag
                tag_logits = tf.reshape(self.tag_logitss, [self.batch_size, -1, self.num_tags])
                return logits, tag_logits 
                
        return logits
            
    def BiGRU(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)
        gru_fw_cell = rnn.GRUCell(self.hidden_size)
        gru_bw_cell = rnn.GRUCell(self.hidden_size)
        outputs,_ = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, self.embedded_words, sequence_length=self.length, dtype=tf.float32)
        gru_out = tf.concat(outputs, axis=2)

        # mask
        mask = tf.cast(self.input_mask, tf.float32)
        gru_out = tf.multiply(gru_out, tf.expand_dims(mask,-1))
        gru_out_pool = tf.reduce_sum(gru_out, axis=1)
        return gru_out, gru_out_pool  
       
    def Ner(self, gru_out):
        batch_size = gru_out.get_shape().as_list()[0]
        gru_out_ = tf.reshape(gru_out, shape=[-1, self.hidden_size*2])
        tag_drop = tf.nn.dropout(gru_out_, keep_prob=self.dropout_keep_prob)
        tag_logit_ = tf.matmul(tag_drop, self.W_tag) + self.b_tag
        tag_logit = tf.reshape(tag_logit_,[self.batch_size, -1, self.num_tags])
        return tag_logit
 
    def attention(self, input_x, num_classes, name=""):
        num_units = input_x.get_shape().as_list()[-1]
        with tf.name_scope(name + "attention"):
            W_transition = tf.Variable(tf.truncated_normal(shape=[self.attention_unit_size, num_units],
                                                           stddev=0.1, dtype=tf.float32), name="W_transition")        
            W_context = tf.Variable(tf.truncated_normal(shape=[num_classes, self.attention_unit_size],
                                                           stddev=0.1, dtype=tf.float32), name="W_context")        
            #attention_matrix = tf.map_fn(
            #    fn=lambda x: tf.matmul(W_context, x),
            #    elems=tf.tanh(
            #        tf.map_fn(
            #            fn=lambda x: tf.matmul(W_transition, tf.transpose(x)),
            #            elems=input_x,
            #            dtype=tf.float32
            #        )
            #    )
            #)
            attention_matrix = tf.tanh(tf.matmul(input_x, tf.transpose(W_transition)))
            attention_matrix = tf.transpose(attention_matrix, perm=[0,2,1])
            attention_matrix = tf.matmul(W_context,attention_matrix) #256*9*8

            # mask
            mask = (1.0 - tf.cast(self.input_mask, tf.float32))*-10000.0
            attention_matrix = attention_matrix + tf.expand_dims(mask, 1)
            # end
            attention_weight = tf.nn.softmax(attention_matrix, name="attention")
            attention_out = tf.matmul(attention_weight, input_x)
            attention_out = tf.reduce_mean(attention_out, axis=1)
        return attention_weight, attention_out


    def local_layer(self, input_x, input_att_weight, num_classes, name=""):
        with tf.name_scope(name+"output"):
            num_units = input_x.get_shape().as_list()[-1]
            W = tf.Variable(tf.truncated_normal(shape=[num_units, num_classes],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[num_classes], dtype=tf.float32), name='b')
            logits = tf.nn.xw_plus_b(input_x, W, b, name="logits")
            scores = tf.sigmoid(logits, name="scores")
            
            # shape of visual: [batch_size, sequence_length]
            visual = tf.multiply(input_att_weight, tf.expand_dims(scores, -1))
            # mask
            mask = (1.0 - tf.cast(self.input_mask, tf.float32))*-10000.0
            visual = visual + tf.expand_dims(mask, 1) 
            visual = tf.nn.softmax(visual)
            visual = tf.reduce_mean(visual, axis=1, name="visual")
            #visual = tf.reduce_sum(visual, axis=1)
            # mask
            mask = (1.0 - tf.cast(self.input_mask, tf.float32))*-10000.0
            visual = visual + mask
            visual = tf.nn.softmax(visual)
        return logits, scores, visual
    
    def local_layer2(self, input_x, num_classes, name=""):
        with tf.name_scope(name+"output"):
            num_units = input_x.get_shape().as_list()[-1]
            W = tf.Variable(tf.truncated_normal(shape=[num_units, num_classes],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[num_classes], dtype=tf.float32), name='b')
            logits = tf.nn.xw_plus_b(input_x, W, b, name="logits")
            scores = tf.sigmoid(logits, name="scores")
            
        return logits, scores

    def fc_layer(self, input_x, name=""):
        with tf.name_scope(name + "fc"):
            num_units = input_x.get_shape().as_list()[-1]
            W = tf.Variable(tf.truncated_normal(shape=[num_units, self.fc_size],
                                                stddev=0.1, dtype=tf.float32, name="W"))
            b = tf.Variable(tf.constant(value=0.1, shape=[self.fc_size],dtype=tf.float32), name="b")
            fc = tf.nn.xw_plus_b(input_x, W, b)
            fc_out = tf.nn.relu(fc)
        return fc_out
 
    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def tag_loss(self):
        log_probs = tf.nn.log_softmax(self.tag_logits, axis=-1)
        one_hot_labels = tf.one_hot(self.input_tag, depth=self.num_tags, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss_tag = tf.reduce_mean(per_example_loss)
        return loss_tag

    def cross_loss(self, labels, logits, weight=None, weight_flag=False, name=''):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits= logits)
        if weight_flag:
            losses = tf.multiply(losses, tf.expand_dims(weight, 0))
        losses = tf.reduce_mean(tf.reduce_sum(losses,axis=1), name=name + "losses")
        #losses = tf.reduce_mean(losses, name=name + "losses")
        return losses
       

    def weighted_cross_loss(self, labels, logits, pos_weight, name):
        losses = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits, pos_weight=pos_weight)
        losses = tf.reduce_mean(losses, name=name + "losses")
        return losses
 
    def l2_loss(self, l2_lambda=0.0001):
        l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables() if 'bias' not in v.name], name="l2_loss") * l2_lambda
        return l2_losses

    def focal_loss(self, labels, logits, name, gamma=1.5, alpha=0.4):
        probs = tf.sigmoid(logits)
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        alpha_ = tf.ones_like(logits)*alpha
        alpha_ = tf.where(labels >0, alpha_, 1.0 - alpha_)
        probs_ = tf.where(labels >0, probs, 1.0 - probs)
        loss_matrix = alpha_ * tf.pow((1.0 - probs_), gamma)
        loss = loss_matrix * ce_loss
        loss = tf.reduce_mean(loss, name=name+"losses")
        return loss


    def visual_loss(self, labels, logits, scores, relation, name):
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        visual_matrix = tf.matmul(scores, relation)
        loss = visual_matrix * ce_loss 
        loss = tf.reduce_mean(loss, name=name+"losses")
        #loss = tf.reduce_sum(loss, name=name+"losses")
        return loss 
    
    def hierarchical_loss(self, labels, logits, labels_before, relation, weight_flag=False, name=''):
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        # another
        #pos_weight = tf.cast(tf.equal(labels_before, 1.0), tf.float32)
        if weight_flag:
            ce_loss = tf.multiply(ce_loss, tf.expand_dims(self.cid3_loss_weight,0)) 
        hierarchical_matrix = tf.matmul(labels_before, relation)
        loss = hierarchical_matrix * ce_loss
        loss = tf.reduce_sum(loss, name=name+"losses") / tf.reduce_sum(hierarchical_matrix) 
        return loss

    def nce_loss_v1(self, weights, biases, labels, inputs, num_sampled, num_classes, name):
        loss = tf.reduce_mean(tf.nn.nce_loss(weights = weights,
                                             biases = biases,
                                             labels = labels,
                                             inputs = inputs,
                                             num_sampled = num_sampled,
                                             num_classes = num_classes,
                                             num_true = 1), name=name+"losses")
        return loss
    def geometric_loss(self, cid2_loss, cid3_loss, product_loss, tag_loss, brand_loss):
        all_loss = tf.div(1.0,5.0) * (tf.math.log(cid2_loss)+tf.math.log(cid3_loss)+tf.math.log(product_loss)+tf.math.log(tag_loss)+tf.math.log(brand_loss)) + tf.div(1.0, 3.0) * (tf.math.log(cid2_loss) + tf.math.log(cid3_loss) + tf.math.log(product_loss))
        return all_loss

    def nce_loss_mask(self, weights, biases, labels, inputs, num_sampled, num_classes, num_true, remove_accidental_hits, div_flag, name):
        loss = nce_loss(weights = weights,
                        biases = biases,
                        labels = labels,
                        inputs = inputs,
                        num_sampled = num_sampled,
                        num_classes = num_classes,
                        num_true = num_true,
                        remove_accidental_hits = remove_accidental_hits,
                        div_flag = div_flag) 
        loss = tf.reduce_mean(loss)
        return loss 
     
        
    def hamming_loss(self, labels, logits, num_label, name):
        probs = tf.sigmoid(logits)
        loss = labels * (1-probs) + (1-labels)*probs
        #loss = tf.reduce_mean(loss, name=name+"losses")
        #loss = tf.reduce_sum(loss, name=name+"losses") / tf.cast(labels.get_shape()[-1], tf.float32)
        loss = tf.reduce_sum(loss, name=name+"losses") / num_label 
        return loss
        
        

    def get_weight(self, sigma, loss):
        sigma2 = tf.div(1.0, 2*sigma*sigma)
        loss_new = tf.add(tf.multiply(sigma2, loss), tf.log(sigma))
        return loss_new

    def multi_task_loss(self):
        loss_first = self.cross_loss(labels=self.input_y_first, logits=self.first_logits, weight=self.cid2_loss_weight, weight_flag=False, name="first_")
        self.loss_first = self.get_weight(self.cid_first_weight, loss_first) 
        
        #loss_second = self.hierarchical_loss(labels=self.input_y_second, logits=self.second_logits, labels_before=self.input_y_first, relation=self.relation_matrix, weight_flag=False, name="second_")
        loss_second = self.cross_loss(labels=self.input_y_second, logits=self.second_logits, name="second_")
        self.loss_second = self.get_weight(self.cid_second_weight, loss_second)

        #loss_product = self.cross_loss(labels=self.input_product, logits=self.product_logits, name="product_")
        #loss_product = self.nce_loss_v1(self.product_w, self.product_b, self.input_product, self.h_drop, 1000, self.num_classes_brand, "product_")
        loss_product = self.nce_loss_mask(self.product_w, self.product_b, self.input_product, self.h_drop, 1000, self.num_classes_product, 15, True, False, "product_")
        self.loss_product = self.get_weight(self.product_weight, loss_product)

        #loss_brand = self.cross_loss(labels=self.input_brand, logits=self.brand_logits, weight=self.brand_loss_weight, weight_flag=True, name="brand_")
        #loss_brand = self.nce_loss(self.brand_w, self.brand_b, self.input_brand, self.h_drop, 1000, self.num_classes_brand, "brand_")
        loss_brand = self.nce_loss_mask(self.brand_w, self.brand_b, self.input_brand, self.h_drop, 2000, self.num_classes_brand, 15, True, False, "brand_")
        self.loss_brand = self.get_weight(self.brand_weight, loss_brand)

        loss_tag = self.tag_loss()
        self.loss_tag = self.get_weight(self.tag_weight, loss_tag)
 
        #losse_second = self.visual_loss(labels=self.input_y_second, logits=self.second_logits, scores=self.first_scores, relation=self.relation_matrix, name="second_")
        #self.losse_brand = self.hamming_loss(labels=self.input_brand, logits=self.brand_logits, num_label=self.num_classes_brand, name='brand_')
        #losse_brand = self.nce_loss(self.brand_w, self.brand_b, self.input_brand, self.h_drop, 1000, self.num_classes_brand, "brand_")
        #l2_losses = self.l2_loss()
        
         
        #loss = tf.add_n([self.loss_first, self.loss_second, self.loss_product, self.loss_tag, self.loss_brand], name="loss")
        loss = tf.reduce_sum(self.loss_first+self.loss_second+self.loss_product+self.loss_brand+self.loss_tag)
        return loss
 
    def train(self):
        """based on the loss, use SGD to update parameter"""
        #self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, tf.train.get_global_step(), self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=None,
                                                   learning_rate=self.decay_learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def train_nodecay(self):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        train_op = opt.minimize(self.loss_val)
        return train_op

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding_projection"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)

        with tf.name_scope("tag_projection"):
            self.W_tag = tf.get_variable("W_tag", shape=[self.hidden_size*2, self.num_tags], initializer=self.initializer)
            self.b_tag = tf.get_variable("b_tag", shape=[self.num_tags])
            
        with tf.name_scope("task_weight"):
            self.cid_first_weight = tf.get_variable("cid_fist_weight", shape=[1], initializer=tf.initializers.random_uniform(minval=0.2, maxval=1))
            self.cid_second_weight = tf.get_variable("cid_second_weight", shape=[1], initializer=tf.initializers.random_uniform(minval=0.2, maxval=1))
            self.tag_weight = tf.get_variable("tag_weight", shape=[1], initializer=tf.initializers.random_uniform(minval=0.2, maxval=1))
            self.product_weight = tf.get_variable("product_weight", shape=[1], initializer=tf.initializers.random_uniform(minval=0.2, maxval=1))
            self.brand_weight = tf.get_variable("brand_weight", shape=[1], initializer=tf.initializers.random_uniform(minval=0.2, maxval=1))
        #with tf.name_scope("attention"):
        #    self.W_w_attention_word = tf.get_variable("W_w_attention_word",shape=[self.hidden_size * 2, self.hidden_size * 2],initializer=self.initializer)
        #    self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[self.hidden_size * 2])
        #    self.context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2],initializer=self.initializer)

