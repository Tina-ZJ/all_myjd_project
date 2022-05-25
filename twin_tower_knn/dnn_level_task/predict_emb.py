# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
import time
tf.reset_default_graph()
import numpy as np
from data_util import load_test,create_term
import os
import codecs
import time
from dnn_model import DNN

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",2,"number of label")
tf.app.flags.DEFINE_integer("first",1,"number of label")
tf.app.flags.DEFINE_integer("pad",8,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint/HAN_300d/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",8,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",200,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("hidden_size",300,"hidden size")
tf.app.flags.DEFINE_string("predict_target_file","./test.txt","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'./test.han',"target file path for final prediction")
tf.app.flags.DEFINE_string("term_index_path","data/term_index.txt",'path of term')


def main(_):
    vocabulary_word2index, vocabulary_index2word= create_term(FLAGS.term_index_path)
    vocab_size = len(vocabulary_word2index)
    testX, lines = load_test(FLAGS.predict_target_file, vocabulary_word2index, FLAGS.pad)

    with tf.Graph().as_default():
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            model = DNN( FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, vocab_size, FLAGS.embed_size, FLAGS.hidden_size, FLAGS.is_training)
            saver=tf.train.Saver()
            if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
                print("Restoring Variables from Checkpoint")
                saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
                #saver.restore(sess,FLAGS.ckpt_dir+'model.ckpt-3')
            else:
                print("Can't find the checkpoint.going to stop")
                return
            number_of_training_data=len(testX);print("number_of_training_data:",number_of_training_data)

            predict_target_file_f = codecs.open(FLAGS.predict_source_file, 'w', 'utf8')
            t0 = time.time()
            for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
                if FLAGS.first==1:
                    sentence =sess.run([model.query_f],feed_dict={model.query:testX[start:end],model.dropout_keep_prob:1})
                else:
                    sentence =sess.run([model.cid_f],feed_dict={model.cid:testX[start:end],model.dropout_keep_prob:1})
                lines_sublist=lines[start:end]
                
                emb_str = [str(x) for x in sentence[0][0]]
                predict_target_file_f.write(' '.join(emb_str)+'\n')
            t1 = time.time()
            print ("all running time: %s " % str(t1-t0))
            predict_target_file_f.close()


if __name__ == "__main__":
    tf.app.run()
