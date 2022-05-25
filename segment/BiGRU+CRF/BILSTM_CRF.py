import math
import helper
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn
from tensorflow.contrib.crf.python.ops import crf
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
import sys
import model_base

reload(sys)
sys.setdefaultencoding('utf8')


class BILSTM_CRF(object):
    def __init__(self, num_chars, num_classes, num_steps=120, num_epochs=3, batch_size=256,
                 emb_dim=300, hidden_dim=200, num_layers=1, dropout_rate=0.5,
                 learning_rate=0.001, embedding_matrix=None, is_training=True):
        # Parameter
        self.bLstm = False
        self.use_peepholes = True
        self.max_f1 = 0
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_chars = num_chars
        self.num_classes = num_classes

        # placeholder of x, y
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [None, self.num_steps])

        # char embedding
        if embedding_matrix is not None:
            embedding = tf.Variable(embedding_matrix, trainable=True, name="emb", dtype=tf.float32)
        else:
            embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])
        inputs_emb = tf.nn.embedding_lookup(embedding, self.inputs)

        # rnn cell
        if self.bLstm:
            rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_dim, use_peepholes=self.use_peepholes)
            rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_dim, use_peepholes=self.use_peepholes)
        else:
            rnn_cell_fw = rnn_cell.GRUCell(self.hidden_dim)
            rnn_cell_bw = rnn_cell.GRUCell(self.hidden_dim)
        # dropout
        if is_training:
            rnn_cell_fw = rnn_cell.DropoutWrapper(rnn_cell_fw, output_keep_prob=(1 - self.dropout_rate))
            rnn_cell_bw = rnn_cell.DropoutWrapper(rnn_cell_bw, output_keep_prob=(1 - self.dropout_rate))


        # get the length of each sample
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)

        # forward and backward
        lstm_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
            [rnn_cell_fw] * num_layers,
            [rnn_cell_bw] * num_layers,
            inputs_emb,
            sequence_length=self.length,
            dtype=tf.float32
        )
        lstm_outputs = tf.reshape(lstm_outputs, [self.batch_size, self.num_steps, self.hidden_dim * 2])

        # softmax
        lstm_outputs = tf.reshape(lstm_outputs, [-1, self.hidden_dim * 2])

        softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.reshape(tf.matmul(lstm_outputs, softmax_w) + softmax_b,
                                 [self.batch_size, self.num_steps, self.num_classes])

        # crf
        self.transition_params = tf.get_variable("transitions", [self.num_classes, self.num_classes])
        log_likelihood, _ = crf.crf_log_likelihood(
            self.logits, self.targets, self.length, self.transition_params)
        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-log_likelihood)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, sess, save_file, x_train, y_train, x_val, y_val):

        log = open('log', 'a')
        # config tensorboard
        tf.summary.scalar("loss",self.loss)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./tensorboard',sess.graph)

        num_iterations = int(math.ceil(1.0 * len(x_train) / self.batch_size))
        cnt = 0
        for epoch in range(self.num_epochs):
            sh_index = np.arange(len(x_train))
            np.random.shuffle(sh_index)
            x_train = x_train[sh_index]
            y_train = y_train[sh_index]
            log.write("current epoch: %d\n" % (epoch))
            for iteration in range(num_iterations):
                x_train_batch, y_train_batch = helper.nextBatch(x_train, y_train,
                                                                start_index=iteration * self.batch_size,
                                                                batch_size=self.batch_size)

                logits, transition_params, loss_train, sequence_lengths, _ = \
                    sess.run([
                        self.logits, self.transition_params, self.loss, self.length, self.train_op],
                        feed_dict={
                            self.inputs: x_train_batch,
                            self.targets: y_train_batch
                        })
                if iteration % 5000 == 0:
                    cnt += 1
                    correct_labels, total_labels = self.evaluate(logits, transition_params, y_train_batch,
                                                                 sequence_lengths)
                    accuracy = 100.0 * correct_labels / float(total_labels)

                    log.write("iteration: %5d, train loss: %.3f, train precision: %.3f%%\n" % (
                        iteration, loss_train, accuracy))
                    log.flush()
                    # print loss_train
                if iteration > 0 and iteration % 5000 == 0:
                    # data to tensorboard
                    feed_dict = {
                        self.inputs: x_train_batch,
                        self.targets: y_train_batch
                    }
                    s = sess.run(merged_summary,feed_dict=feed_dict)
                    writer.add_summary(s,iteration)

                    self.verify(sess, x_val, y_val, log, save_file, epoch, iteration)

            self.verify(sess, x_val, y_val, log, save_file, epoch, iteration)
        log.close()


    def verify(self, sess, x_val, y_val, log, save_file, epoch, iteration, sample=False):
        if sample:
            numValIterations = int(math.log(1.0 * len(x_val) / self.batch_size + 2.0, 2))
        else:
            numValIterations = int(math.ceil(1.0 * len(x_val) / self.batch_size))
        numCorrects = 0
        numTotal = 0
        for valIte in range(numValIterations):
            if sample:
                X_val_batch, y_val_batch = helper.nextRandomBatch(x_val, y_val,
                                                                  batch_size=self.batch_size)
            else:
                X_val_batch, y_val_batch = helper.nextBatch(x_val, y_val,
                                                            start_index=valIte * self.batch_size,
                                                            batch_size=self.batch_size)
            logits, transition_params, loss_train, sequence_lengths = \
                sess.run([
                    self.logits, self.transition_params, self.loss, self.length],
                    feed_dict={
                        self.inputs: X_val_batch,
                        self.targets: y_val_batch
                    })
            correct_labels, total_labels = self.evaluate(logits, transition_params, y_val_batch, sequence_lengths)
            numCorrects += correct_labels
            numTotal += total_labels
        if numTotal > 0:
            saver = tf.train.Saver()
            accuracy = 100.0 * numCorrects / float(numTotal)
            log.write("epoch: %5d, iteration: %5d, verify precision: %.3f%%\n" % (epoch, iteration, accuracy))
            log.flush()
            if accuracy >= self.max_f1:
                self.max_f1 = accuracy
                saver.save(sess, save_file)
                info = open(save_file + 'log', 'a+')
                info.write("epoch: %5d, iteration: %5d, verify precision: %.3f%%\n" % (
                    epoch, iteration, accuracy
                ))
                info.flush()
                info.close()

    def evaluate(self, tf_unary_scores, tf_transition_params, y, sequence_lengths):
        correct_labels = 0
        total_labels = 0
        for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, y,
                                                          sequence_lengths):
            if sequence_length_ > 0:
                # Remove padding from the scores and tag sequence.
                tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                y_ = y_[:sequence_length_]

                # Compute the highest scoring sequence.
                viterbi_sequence, _ = crf.viterbi_decode(
                    tf_unary_scores_, tf_transition_params)

                # Evaluate word-level accuracy.
                correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                total_labels += sequence_length_

        return correct_labels, total_labels

    def decode(self, sess, sentence, pretags=None):
        logits, transition_params, sequence_lengths = \
            sess.run([
                self.logits, self.transition_params, self.length],
                feed_dict={
                    self.inputs: sentence,
                })
        score = logits[0][:sequence_lengths[0]]
        if sequence_lengths[0] <= 0:
            return []
        if pretags != None:
            plen = min(len(pretags), self.num_steps)
            for i in range(plen):
                if pretags[i] >= 0:
                    score[i][pretags[i]] += 10.0
        viterbi_sequence, _ = crf.viterbi_decode(score, transition_params)
        return viterbi_sequence

    def predictBatch(self, sess, x):
        results = []
        logits, transition_params, sequence_lengths = \
            sess.run([
                self.logits, self.transition_params, self.length],
                feed_dict={
                    self.inputs: x,
                })
        for logit, _seq_len in zip(logits, sequence_lengths):
            if _seq_len > 0:
                score = logit[:_seq_len]
                viterbi_sequence, _ = crf.viterbi_decode(score, transition_params)
                results.append(viterbi_sequence)
            else:
                results.append([])
        return results
