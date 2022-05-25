# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
tf.reset_default_graph()
import numpy as np
from data_util import create_term
import os
import codecs
from dnn_model import DNN

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat
import argparse

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("ckpt","./checkpoint/dnn","checkpoint location for the model")
tf.app.flags.DEFINE_string("pb_path","./model_v1","checkpoint location for the model")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_integer("embed_size",200,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("hidden_size",300,"hidden size")
tf.app.flags.DEFINE_string("term_index_path","data/term_index.txt",'path of term')


def export_pb_model():
    vocabulary_word2index, vocabulary_index2word= create_term(FLAGS.term_index_path)
    vocab_size = len(vocabulary_word2index)
    with tf.Graph().as_default():
        model = DNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, vocab_size, FLAGS.embed_size, FLAGS.hidden_size, FLAGS.is_training)
        model_signature = signature_def_utils.build_signature_def(
            inputs={
                "terms_ids": utils.build_tensor_info(model.input_x),
                "keep_prob_hidden": utils.build_tensor_info(model.dropout_keep_prob)
            },
            outputs={
                "prediction": utils.build_tensor_info(model.predictions),
                "tag_predictions": utils.build_tensor_info(model.tag_predictionss),
                "attention": utils.build_tensor_info(model.attention)
            },
            method_name=signature_constants.CLASSIFY_METHOD_NAME
        )

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )

        sess = tf.Session(config=session_conf)
        saver = tf.train.Saver()
        #saver.restore(sess,_ckpt+'model.ckpt-6')
        print(tf.train.latest_checkpoint(FLAGS.ckpt))
        saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt))
    
        builder = saved_model_builder.SavedModelBuilder(FLAGS.pb_path)
        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                'cat_han_model_signature':
                    model_signature,
            }
        )
    
        builder.save()
    



if __name__ == "__main__":
    export_pb_model()
