# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import numpy as np
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn

class HierarchicalAttention:
    def __init__(self, normalize, gamma, alpha, loss_number, num_count, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, char_length, num_sentences,
                 vocab_size, char_size, embed_size,hidden_size, is_training, multi_label_flag=True,char_attention_flag=True, count_flag=False,initializer=initializers.xavier_initializer(),clip_gradients=5.0):
        """init all hyperparameter here"""
        # set hyperparamter
        self.normalize = normalize
        self.gamma = gamma
        self.alpha = alpha
        self.loss_number = loss_number
        self.num_count = num_count
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.char_length = char_length
        self.num_sentences = num_sentences
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.95)
        self.initializer = initializer
        self.multi_label_flag = multi_label_flag
        self.char_attention_flag = char_attention_flag
        self.count_flag = count_flag
        self.hidden_size = hidden_size
        self.clip_gradients=clip_gradients

        # add placeholder (X,label)
        #self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        # sequence true length
        self.length = tf.reduce_sum(tf.sign(self.input_x), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)

        self.input_mask = tf.cast(tf.not_equal(self.input_x, 0), tf.float32)
        if self.char_attention_flag: 
            #self.input_char = tf.placeholder(tf.int32, [None, self.char_length], name="input_char")
            self.input_char = tf.placeholder(tf.int32, [None, None], name="input_char")
            self.input_mask_char = tf.cast(tf.not_equal(self.input_char, 0), tf.float32)

        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes],name="input_y_multilabel")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference_simple()
        self.predictions = tf.reshape(tf.sigmoid(self.logits),[-1, self.num_classes], name='prediction')


        if not self.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        else:
            self.tp = tf.reduce_sum(tf.cast(tf.greater(self.predictions, 0.3), tf.float32) * tf.cast(tf.equal(self.input_y_multilabel, 1), tf.float32))
            self.tn = tf.reduce_sum(tf.cast(tf.less(self.predictions, 0.3), tf.float32) * tf.cast(tf.equal(self.input_y_multilabel, 0), tf.float32))
            self.fp = tf.reduce_sum(tf.cast(tf.greater(self.predictions, 0.3), tf.float32) * tf.cast(tf.equal(self.input_y_multilabel, 0), tf.float32))
            self.fn = tf.reduce_sum(tf.cast(tf.less(self.predictions, 0.3), tf.float32) * tf.cast(tf.equal(self.input_y_multilabel, 1), tf.float32))
            self.accuracy = tf.div(self.tp+self.tn, self.tp+self.tn+self.fp+self.fn, name='accuracy')
            self.precision = tf.div(self.tp, self.tp+self.fp, name='precision')
            self.recall = tf.div(self.tp, self.tp+self.fn, name='recall')
            self.f1 = tf.div(2*self.precision*self.recall, self.precision+self.recall, name='F1')


        if multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()

        if not is_training:
            return
        if self.count_flag:
            self.predictions_count, self.lcnt_loss, self.acc = self.loss_count()
            self.count_train_op = self.train_count()
        else:
            self.train_op = self.train()
        
        # tensorboard
        tf.summary.scalar("loss", self.loss_val)
        tf.summary.scalar("precision", self.precision)
        tf.summary.scalar("recall", self.recall)
        tf.summary.scalar("F1", self.f1)
        tf.summary.scalar("larning_rate", self.decay_learning_rate)
        self.summary_merge = tf.summary.merge_all()

    def attention_char_level(self, hidden_state):
        """
        input1:self.hidden_state: hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        input2:sentence level context vector:[batch_size*num_sentences,hidden_size*2]
        :return:representation.shape:[batch_size*num_sentences,hidden_size*2]
        """
        hidden_state_ = tf.stack(hidden_state, axis=1)
        # 0) one layer of feed forward network
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1,self.hidden_size * 2])

        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2, self.W_w_attention_char) + self.W_b_attention_char)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.char_length, self.hidden_size * 2])

        # 1) get logits for each char in the sentence.
        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vecotor_char)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,axis=2)

        ### Mask #####
        attention_logits = tf.multiply(self.input_mask_char, attention_logits)
        

        # 2) get possibility distribution for each word in the sentence.
        #p_attention = tf.nn.softmax(attention_logits - attention_logits_max)

        p_attention = tf.nn.softmax(attention_logits)
        # 3) get weighted hidden state by attention vector

        ## Mask####
        p_attention = tf.multiply(self.input_mask_char, p_attention)
        p_attention_sum = tf.reduce_sum(p_attention, axis=1)
        p_attention_sum = tf.expand_dims(p_attention_sum, -1)
        ## normalization to sum 1
        p_attention = tf.divide(p_attention, p_attention_sum)

        p_attention_expanded = tf.expand_dims(p_attention, axis=2)
        sentence_representation = tf.multiply(p_attention_expanded,hidden_state_)
        sentence_representation = tf.reduce_sum(sentence_representation,axis=1)
        return p_attention, sentence_representation

    def attention_word_level(self, hidden_state):
        """
        input1:self.hidden_state: hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        input2:sentence level context vector:[batch_size*num_sentences,hidden_size*2]
        :return:representation.shape:[batch_size*num_sentences,hidden_size*2]
        """
        hidden_state_ = tf.stack(hidden_state, axis=1)
        # 0) one layer of feed forward network
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1,self.hidden_size * 2])

        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,self.W_w_attention_word) + self.W_b_attention_word)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.sequence_length,self.hidden_size * 2])


        # 1) get logits for each word in the sentence.
        hidden_state_context_similiarity = tf.multiply(hidden_representation,self.context_vecotor_word)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,axis=2)
        ### Mask #####
        attention_logits = tf.multiply(self.input_mask, attention_logits)
        
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
        #attention_logits_max = tf.reduce_max(attention_logits, axis=1,keep_dims=True)
        # 2) get possibility distribution for each word in the sentence.
        #p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
        p_attention = tf.nn.softmax(attention_logits)
        # 3) get weighted hidden state by attention vector

        ## Mask####
        p_attention = tf.multiply(self.input_mask, p_attention)
        p_attention_sum = tf.reduce_sum(p_attention, axis=1)
        p_attention_sum = tf.expand_dims(p_attention_sum, -1)
        p_attention = tf.divide(p_attention, p_attention_sum)

        p_attention_expanded = tf.expand_dims(p_attention, axis=2)

        sentence_representation = tf.multiply(p_attention_expanded,hidden_state_)
        sentence_representation = tf.reduce_sum(sentence_representation,
                                                axis=1)
        return p_attention, sentence_representation

    def attention_sentence_level(self, hidden_state_sentence):
        """
        input1: hidden_state_sentence: a list,len:num_sentence,element:[None,hidden_size*4]
        input2: sentence level context vector:[self.hidden_size*2]
        :return:representation.shape:[None,hidden_size*4]
        """
        hidden_state_ = tf.stack(hidden_state_sentence, axis=1)

        # 0) one layer of feed forward
        hidden_state_2 = tf.reshape(hidden_state_,shape=[-1, self.hidden_size * 4])
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,self.W_w_attention_sentence) + self.W_b_attention_sentence)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.num_sentences,self.hidden_size * 2])
        # 1) get logits for each word in the sentence.
        hidden_state_context_similiarity = tf.multiply(hidden_representation,self.context_vecotor_sentence)  # shape:[None,num_sentence,hidden_size*2]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,axis=2)
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)  # shape:[None,1]
        # 2) get possibility distribution for each word in the sentence.
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)  # shape:[None,num_sentence]
        # 3) get weighted hidden state by attention vector(sentence level)
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # shape:[None,num_sentence,1]
        sentence_representation = tf.multiply(p_attention_expanded,hidden_state_)
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)
        return  p_attention, sentence_representation

    def inference_simple(self):
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
        p_attention = tf.nn.softmax(uv-tf.reduce_max(uv,-1,keepdims=True))

        ## Mask attention and to sum(1)##
        p_attention = tf.multiply(self.input_mask, p_attention)
        p_attention_sum = tf.reduce_sum(p_attention, axis=1)
        p_attention_sum = tf.expand_dims(p_attention_sum, -1)
        self.attention = tf.divide(p_attention, p_attention_sum)
        
        ## output ## 
        sentence_representation = tf.reduce_sum(self.hidden_state*tf.expand_dims(self.attention, -1), 1) 

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(sentence_representation,keep_prob=self.dropout_keep_prob)

        # 5. logits
        with tf.name_scope("output"):
                logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
        return logits
            
        
    def inference(self):
        """main computation graph here: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier"""
        # 1.Word Encoder
        # 1.1 embedding of words
        input_x = tf.split(self.input_x, self.num_sentences,axis=1)
        input_x = tf.stack(input_x, axis=1)  # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,input_x)
        embedded_words_reshaped = tf.reshape(self.embedded_words, shape=[-1, self.sequence_length,self.embed_size])
        if self.char_attention_flag:
            input_char = tf.split(self.input_char, self.num_sentences,axis=1)
            input_char = tf.stack(input_char, axis=1)  # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
            self.embedded_chars = tf.nn.embedding_lookup(self.CharEmbedding,input_char)
            embedded_chars_reshaped = tf.reshape(self.embedded_chars, shape=[-1, self.char_length,self.embed_size])
        # 1.2 forward  and backward gru
        hidden_state_forward_list = self.gru_forward_word_level(embedded_words_reshaped, self.sequence_length)
        hidden_state_backward_list = self.gru_backward_word_level(embedded_words_reshaped, self.sequence_length)
        self.hidden_state = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in zip(hidden_state_forward_list, hidden_state_backward_list)]
        self.attention, sentence_representation_word = self.attention_word_level(self.hidden_state)

        if self.char_attention_flag:
            hidden_state_forward_list_char = self.gru_forward_word_level(embedded_chars_reshaped, self.char_length)
            hidden_state_backward_list_char = self.gru_backward_word_level(embedded_chars_reshaped, self.char_length)
            self.hidden_state_char = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in zip(hidden_state_forward_list_char, hidden_state_backward_list_char)]
            self.attention_char, sentence_representation_char = self.attention_char_level(self.hidden_state_char)

        ### concat ######
        if self.char_attention_flag:
            sentence_representation = tf.concat(axis=1, values=[sentence_representation_word, sentence_representation_char])
        else:
            sentence_representation = sentence_representation_word

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(sentence_representation,keep_prob=self.dropout_keep_prob)

        # 5. logits
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
        return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def ohnm(self):
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
        pos_weight = tf.cast(tf.equal(self.input_y_multilabel,1),tf.float32)
        neg_weight = 1 - pos_weight
        n_pos = tf.reduce_sum(pos_weight)
        n_neg = tf.reduce_sum(neg_weight)
        n_selected = tf.minimum(n_pos*3, n_neg)
        n_selected = tf.cast(tf.maximum(n_selected,1),tf.int32)
        neg_mask = tf.equal(self.input_y_multilabel, 0)
        neg_score = tf.where(neg_mask, ce_loss, tf.zeros_like(ce_loss))
        vals,_ = tf.nn.top_k(neg_score, k=n_selected)
        idx = tf.arg_min(vals[-1],0)
        min_score = vals[-1][idx]
        selected_neg_mask = tf.logical_and(neg_score>=min_score, neg_mask)
        neg_weight = tf.cast(selected_neg_mask, tf.float32)
        loss_weight = pos_weight + neg_weight
        loss = tf.reduce_sum(ce_loss * loss_weight) / tf.reduce_sum(loss_weight)
        return loss


    def focal_loss(self, gamma, alpha, normalize=True):
        probs = tf.sigmoid(self.logits)
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
        alpha_ = tf.ones_like(self.logits)*alpha
        alpha_ = tf.where(self.input_y_multilabel >0, alpha_, 1.0 - alpha_)
        probs_ = tf.where(self.input_y_multilabel >0, probs, 1.0 - probs)
        
        loss_matrix = alpha_ * tf.pow((1.0 - probs_), gamma)
        loss = loss_matrix * ce_loss
        loss = tf.reduce_sum(loss)
        if normalize:
            n_pos = tf.reduce_sum(self.input_y_multilabel)
            total_weights = tf.stop_gradient(tf.reduce_sum(loss_matrix))
            total_weights = tf.Print(total_weights, [n_pos, total_weights])
            def has_pos():
                return loss / tf.cast(n_pos, tf.float32)
            def no_pos():
                return loss
            loss = tf.cond(n_pos >0, has_pos, no_pos)
        return loss

    def loss_multilabel(self, l2_lambda=0.00001*10, e=0.001): #0.00001
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]
            if self.loss_number==0:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
            elif self.loss_number==1:
                losses = tf.nn.weighted_cross_entropy_with_logits(targets=self.input_y_multilabel,logits=self.logits,pos_weight=2)
            elif self.loss_number ==2:
                losses = self.focal_loss(self.gamma, self.alpha, self.normalize) 
            elif self.loss_number==3:
                losses = (1-e)*tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)+e*tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.input_y_multilabel)/self.num_classes,logits=self.logits)
            elif self.loss_number ==4:
                losses = self.ohnm()
            else:
                losses = tf.losses.mean_pairwise_squared_error(labels=self.input_y_multilabel, predictions=tf.sigmoid(self.logits))
            
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)

            print("sigmoid_cross_entropy_with_logits.losses:", losses)
            if self.loss_number==0 or self.loss_number==1 or self.loss_number==3:
                losses = tf.reduce_sum(losses, axis=1)
            if self.loss_number==2 or self.loss_number==4:
                loss = losses
            else:
                loss = tf.reduce_mean(losses)
            #l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            #loss = loss + l2_losses
        return loss

    def loss_count(self):
        num_bins = self.num_count
        sentence_level_output = tf.stop_gradient(self.h_drop)
        cnt_h1 = tf.layers.dense(sentence_level_output, 1024, activation=tf.nn.relu)
        cnt_h2 = tf.layers.dense(cnt_h1, 512, activation=tf.nn.relu)
        cnt_h3 = tf.layers.dense(cnt_h2, 256, activation=tf.nn.relu)
        lcnt = tf.layers.dense(cnt_h3, self.num_count, activation=tf.nn.relu)
        predictions_count = tf.argmax(lcnt, axis=1, name='prediction_count')
        label_count = tf.reduce_sum(self.input_y_multilabel,1)
        tails = num_bins*tf.ones_like(label_count)
        bins = tf.where(label_count > num_bins, tails, label_count)
        labels = bins -1
        labels = tf.cast(labels, tf.int64)

        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lcnt, labels=labels)
        lcnt_loss = tf.reduce_mean(xent, name ='count_loss')

        correct_pred = tf.equal(labels,predictions_count)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return predictions_count, lcnt_loss, acc


    def train(self):
        """based on the loss, use SGD to update parameter"""
        self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=self.decay_learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def train_count(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        count_train_op = tf_contrib.layers.optimize_loss(self.lcnt_loss, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return count_train_op



    def gru_single_step_word_level(self, Xt, h_t_minus_1):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size*num_sentences,embed_size]
        :param h_t_minus_1:[batch_size*num_sentences,embed_size]
        :return:
        """
        # update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1,self.U_z) + self.b_z)  # z_t:[batch_size*num_sentences,self.hidden_size]
        # reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1,self.U_r) + self.b_r)  # r_t:[batch_size*num_sentences,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) +r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)  # h_t_candiate:[batch_size*num_sentences,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t

    def gru_single_step_sentence_level(self, Xt,h_t_minus_1):
        """
        single step of gru for sentence level
        :param Xt:[batch_size, hidden_size*2]
        :param h_t_minus_1:[batch_size, hidden_size*2]
        :return:h_t:[batch_size,hidden_size]
        """
        # update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_sentence) + tf.matmul(h_t_minus_1,self.U_z_sentence) + self.b_z_sentence)
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_sentence) + tf.matmul(h_t_minus_1, self.U_r_sentence) + self.b_r_sentence)
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_sentence) + r_t * (tf.matmul(h_t_minus_1, self.U_h_sentence)) + self.b_h_sentence)
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t

    # forward gru for first level: word levels
    def gru_forward_word_level(self, embedded_words, length):
        """
        :param embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return:forward hidden state: a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = tf.split(embedded_words, length,axis=1)
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]
        h_t = tf.ones((self.batch_size * self.num_sentences,self.hidden_size))
        h_t_forward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt,h_t)
            h_t_forward_list.append(h_t)
        return h_t_forward_list

    # backward gru for first level: word level
    def gru_backward_word_level(self, embedded_words, length):
        """
        :param   embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return: backward hidden state:a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = tf.split(embedded_words, length,axis=1)
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]
        embedded_words_squeeze.reverse()
        h_t = tf.ones((self.batch_size * self.num_sentences, self.hidden_size))
        h_t_backward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_backward_list.append(h_t)
        h_t_backward_list.reverse()
        return h_t_backward_list

    # forward gru for second level: sentence level
    def gru_forward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [batch_size,hidden_size]
        """
        # split embedded_words
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences, axis=1)
        sentence_representation_squeeze = [tf.squeeze(x, axis=1) for x in sentence_representation_splitted]
        h_t = tf.ones((self.batch_size, self.hidden_size * 2))  # TODO
        h_t_forward_list = []
        for time_step, Xt in enumerate(sentence_representation_squeeze):  # Xt:[batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level(Xt,h_t)
            h_t_forward_list.append(h_t)
        return h_t_forward_list

    # backward gru for second level: sentence level
    def gru_backward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [batch_size,hidden_size]
        """
        # split embedded_words
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences,axis=1)
        sentence_representation_squeeze = [tf.squeeze(x, axis=1) for x in sentence_representation_splitted]
        sentence_representation_squeeze.reverse()
        h_t = tf.ones((self.batch_size, self.hidden_size * 2))
        h_t_forward_list = []
        for time_step, Xt in enumerate(sentence_representation_squeeze):
            h_t = self.gru_single_step_sentence_level(Xt,h_t)
            h_t_forward_list.append(h_t)
        h_t_forward_list.reverse()
        return h_t_forward_list

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding_projection"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)
            if self.char_attention_flag:
                self.CharEmbedding = tf.get_variable("CharEmbedding", shape=[self.char_size, self.embed_size],initializer=self.initializer)
                self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 4, self.num_classes],initializer=self.initializer)
            else:
                self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 2, self.num_classes],initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

        # GRU parameters:update gate related
        with tf.name_scope("gru_weights_word_level"):
            self.W_z = tf.get_variable("W_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_z = tf.get_variable("U_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
            # GRU parameters:reset gate related
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_r = tf.get_variable("U_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])

            self.W_h = tf.get_variable("W_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_h = tf.get_variable("U_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])

        # with tf.name_scope("gru_weights_sentence_level"):
        #     self.W_z_sentence = tf.get_variable("W_z_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
        #                                         initializer=self.initializer)
        #     self.U_z_sentence = tf.get_variable("U_z_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
        #                                         initializer=self.initializer)
        #     self.b_z_sentence = tf.get_variable("b_z_sentence", shape=[self.hidden_size * 2])
        #     # GRU parameters:reset gate related
        #     self.W_r_sentence = tf.get_variable("W_r_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
        #                                         initializer=self.initializer)
        #     self.U_r_sentence = tf.get_variable("U_r_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
        #                                         initializer=self.initializer)
        #     self.b_r_sentence = tf.get_variable("b_r_sentence", shape=[self.hidden_size * 2])
        #
        #     self.W_h_sentence = tf.get_variable("W_h_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
        #                                         initializer=self.initializer)
        #     self.U_h_sentence = tf.get_variable("U_h_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
        #                                         initializer=self.initializer)
        #     self.b_h_sentence = tf.get_variable("b_h_sentence", shape=[self.hidden_size * 2])

        with tf.name_scope("attention"):
            self.W_w_attention_word = tf.get_variable("W_w_attention_word",shape=[self.hidden_size * 2, self.hidden_size * 2],initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[self.hidden_size * 2])
            self.context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2],initializer=self.initializer)
            if self.char_attention_flag:
                self.W_w_attention_char = tf.get_variable("W_w_attention_char",shape=[self.hidden_size * 2, self.hidden_size * 2],initializer=self.initializer)
                self.W_b_attention_char = tf.get_variable("W_b_attention_char", shape=[self.hidden_size * 2])
                self.context_vecotor_char = tf.get_variable("what_is_the_informative_char", shape=[self.hidden_size * 2],initializer=self.initializer)


