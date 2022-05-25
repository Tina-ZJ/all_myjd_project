# -*- coding: utf-8 -*-
#import sys
import tensorflow as tf
import re
import numpy as np
from model import Semantic
from data_util import create_term
import collections
import tensorflow.contrib as tf_contrib
import os
import codecs
import traceback
from read_file import run_shell_cmd

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_cpu_threads",30,"number of cpu")
tf.app.flags.DEFINE_float("keep_prob",0.5,"keep_prob")  
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")   #0.001 
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.app.flags.DEFINE_integer("decay_steps", 32000, "how many steps before decay learning rate.")  #24000
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("margin", 1.6, "triple loss margin.")
tf.app.flags.DEFINE_integer("query_len",8,"max query length")   
tf.app.flags.DEFINE_integer("item_len",30,"max item length")   
tf.app.flags.DEFINE_integer("term_dim",200,"term embedding size")  
tf.app.flags.DEFINE_integer("hidden_size",256,"hidden size")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")  # 6 10 12 8 3 
tf.app.flags.DEFINE_string("term_path","data/term_index.txt",'path of term')
tf.app.flags.DEFINE_integer("train_sample_num",195635677,"train sample num")
tf.app.flags.DEFINE_integer("dev_sample_num",1327682,"dev sample num")
tf.app.flags.DEFINE_boolean("do_train",True,"whether to run training")
tf.app.flags.DEFINE_boolean("do_eval",False,"whether to run eval on the dev")
tf.app.flags.DEFINE_boolean("do_predict",False,"whether to run model in inference")
tf.app.flags.DEFINE_boolean("sub_flag",False,"whether to run model in inference")
tf.app.flags.DEFINE_boolean("use_onehot_emb",False,"whether to use one hot embedding to extract")
tf.app.flags.DEFINE_boolean("is_training",True,"whether to train")
tf.app.flags.DEFINE_string("output_dir","./output_0.001_1.6/checkpoint",'save model path')
tf.app.flags.DEFINE_string("init_checkpoint",None,'init  model parameter from pretrain model')
tf.app.flags.DEFINE_integer("save_checkpoints_steps",10000, "how many steps to make estimator call")


command = "hadoop fs -ls hdfs://ns1013/user/recsys/suggest/app.db/qp_common_file/qp_semantic/2021-09-20/v2/ | grep part | awk '{print $NF}' 2>/dev/null"

train_sample_file = run_shell_cmd(command)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    assignment_map = {}
    initialized_variable_names = {}
    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    
    ckpt_file = tf.train.latest_checkpoint(init_checkpoint)
    init_vars = tf.train.list_variables(ckpt_file)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable or name in ("medicine_class/W_medicine_class", "medicine_class/b_medicine_class"):
            continue
        #assignment_map[name] = name
        assignment_map[name] = name_to_variable[name] 
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
    return (assignment_map, initialized_variable_names)
        


def model_fn_builder(init_checkpoint, term_size):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(" name = %s, shape = %s" % (name, features[name].shape))

        # input features
        input_query = features["keyword_ids"]
        pos_item = features["pos_ids"]
        neg_item = features["neg_ids"]

        #is_training = tf.cast((mode == tf.estimator.ModeKeys.TRAIN), tf.bool, name='is_training')

        model = Semantic(
                input_query = input_query,
                pos_item = pos_item,
                neg_item = neg_item,
                margin = FLAGS.margin,
                keep_prob = FLAGS.keep_prob,
                learning_rate = FLAGS.learning_rate,
                decay_steps = FLAGS.decay_steps,
                decay_rate = FLAGS.decay_rate,
                batch_size = FLAGS.batch_size,
                vocab_size = term_size,
                embed_size = FLAGS.embed_size,
                hidden_size = FLAGS.hidden_size,
                is_training = FLAGS.is_training)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
       
        # init from exist model 
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint) 
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
       
        # print train vars 
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("    name = %s, shape = %s%s", var.name, var.shape, init_string)
   
        # show result 
        if mode == tf.estimator.ModeKeys.TRAIN:
            loss, train_op = model.loss, model.train_op
            pos_acc, neg_acc = model.pos_acc, model.neg_acc 
            global_step = tf.train.get_or_create_global_step()
            # log 
            logged_tensors = {
            "global_step": global_step,
            "loss": loss,
            "pos_acc": pos_acc,
            "neg_acc": neg_acc,
                }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    scaffold_fn=None,
                    training_hooks=[tf.train.LoggingTensorHook(logged_tensors, every_n_iter=2000)])
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(loss):
                return {
                        "loss": loss,
                        }
                eval_metrics = (metric_fn, [loss])
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode,
                        loss=loss,
                        eval_metrics=eval_metrics,
                        scaffold_fn=None)
        else:
            output_sec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={"predictions": model.predictions},
                                  scaffold_fn=None)
        return output_spec
    return model_fn


def file_based_input_fn_builder(num_cpu_threads, input_file, batch_size, query_len, item_len, is_training, drop_remainder):
    name_to_features = {
        'keyword_ids': tf.FixedLenFeature([query_len], tf.int64),
        'pos_ids': tf.FixedLenFeature([item_len], tf.int64),
        'neg_ids': tf.FixedLenFeature([item_len], tf.int64)
    }

    def pad_or_trunc(t):
        k = seq_len
        dim = tf.size(t)
        return tf.cond(tf.equal(dim, k), lambda:t, lambda: tf.cond(tf.greater(dim, k), lambda: tf.slice(t, 0, k), lambda: tf.concat([t, tf.zeros(k-dim, dtype=tf.int32)], 0)))


    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        #for name in list(example.keys()):
        #    t = example[name]
        #    if t.dtype == tf.int64:
        #        t = tf.to_int32(t)
        #    example[name] = t

        #one_hot_enc = tf.one_hot(indices=example["cid3_idx"], depth=FLAGS.num_classes_second)
        #example["cid3_idx"] = tf.reduce_sum(one_hot_enc, axis=0)

        return example

    def input_fn(params):
        """ The actual input function. """
        if  is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_file))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_file))
            
            cycle_length = min(num_cpu_threads, len(input_file))
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=25600)
        else:
            d = tf.data.TFRecordDataset(input_file)
            d = d.repeat()
 
        d = d.apply(
            tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size = batch_size,
                    num_parallel_batches=num_cpu_threads,
                    drop_remainder=drop_remainder))
        return d
    return input_fn
 
            
        
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of do_train, do_eval or do_predict must be True")

    tpu_cluster_resolver = None
    is_per_host =  tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=None,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    
    #1. load term
    word2idx, idx2word= create_term(FLAGS.term_path)
    term_size = max(idx2word.keys())+1
    print("term_size:",term_size)
  
    # load bert config
    model_fn = model_fn_builder(
                init_checkpoint=FLAGS.init_checkpoint, 
                term_size=term_size) 
   
    # init from checkpoint: restore from the original checkpoint
    if FLAGS.sub_flag:
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from='./output_init/checkpoint')
    else:
        ws = None 
    estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=None,
                model_fn=model_fn,
                warm_start_from=ws,
                config=run_config,
                train_batch_size=FLAGS.batch_size,
                eval_batch_size=FLAGS.batch_size,
                predict_batch_size=FLAGS.batch_size)

    if FLAGS.do_train:
        num_train_steps = int(FLAGS.train_sample_num/FLAGS.batch_size)*FLAGS.num_epochs
        print("*****all steps **************", num_train_steps) 
        train_input_fn = file_based_input_fn_builder(
                        num_cpu_threads=FLAGS.num_cpu_threads,
                        input_file=train_sample_file,
                        batch_size=FLAGS.batch_size,
                        query_len=FLAGS.query_len,
                        item_len=FLAGS.item_len, 
                        is_training=True,
                        drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

if __name__ == "__main__":
    tf.app.run()
