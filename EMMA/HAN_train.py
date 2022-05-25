# -*- coding: utf-8 -*-
#import sys
import tensorflow as tf
import numpy as np
from HAN_model import HierarchicalAttention

from data_util import create_term
#from tflearn.data_utils import  pad_sequences
import os
import codecs
#import batch_read_tfrecord
import common
import traceback
from read_file import run_shell_cmd

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("attention_unit_size",100,"transition unit")
tf.app.flags.DEFINE_integer("num_cpu_threads",30,"number of cpu")
tf.app.flags.DEFINE_float("alpha",0.5,"global and local weight")
tf.app.flags.DEFINE_float("keep_prob",0.5,"keep_prob")
tf.app.flags.DEFINE_float("threshold",0.3,"predict threshold")
tf.app.flags.DEFINE_integer("num_classes",4939,"number of label")
tf.app.flags.DEFINE_integer("num_classes_first",4939,"number of cid")
tf.app.flags.DEFINE_integer("num_classes_second",4939,"number of cid2")
tf.app.flags.DEFINE_integer("num_classes_third",4939,"number of cid3")
tf.app.flags.DEFINE_integer("num_classes_product",4939,"number of product")
tf.app.flags.DEFINE_integer("num_classes_brand",4939,"number of brand")
tf.app.flags.DEFINE_integer("num_tags",10,"number of tags")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 100000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_integer("sequence_length",15,"max sentence length")
tf.app.flags.DEFINE_integer("first_length",10,"max first y length")
tf.app.flags.DEFINE_integer("second_length",15,"max second y length")
tf.app.flags.DEFINE_integer("third_length",15,"max third y length")
tf.app.flags.DEFINE_integer("embed_size",200,"embedding size")  #200
tf.app.flags.DEFINE_integer("hidden_size",200,"hidden size")   #128
tf.app.flags.DEFINE_integer("fc_size",512,"hidden size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",6,"number of epochs to run.")
tf.app.flags.DEFINE_string("train_sample_file","data/train_sample.tfrecord","path of traning data.")
tf.app.flags.DEFINE_string("dev_sample_file","data/dev_sample.tfrecord","path of dev data.")
tf.app.flags.DEFINE_string("term_index_path","data/term_index.txt",'path of term')
tf.app.flags.DEFINE_string("cid3_path","data/cid3_name.txt",'path of cid3')
tf.app.flags.DEFINE_string("cid2_path","data/cid2_name.txt",'path of cid2')
tf.app.flags.DEFINE_string("cid_path","data/cid_name.txt",'path of cid')
tf.app.flags.DEFINE_string("cid2_weight","data/cid2.weight",'path of cid2 weight')
tf.app.flags.DEFINE_string("cid3_weight","data/cid3.weight",'path of cid3 weight')
tf.app.flags.DEFINE_string("brand_weight","data/brand.weight",'path of brand weight')
tf.app.flags.DEFINE_boolean("weight_flag",False,"whether augment tail label")
tf.app.flags.DEFINE_boolean("relation_flag",False,"whether use relation flag")
tf.app.flags.DEFINE_string("cid2_cid3_relation","data/cid2_cid3.weight",'path of cid2 and cid3 relation')
tf.app.flags.DEFINE_string("cid3_product_relation","data/cid3_product.weight",'path of product and cid3 relation')
tf.app.flags.DEFINE_string("cid3_brand_relation","data/cid3_brand.weight",'path of product and cid3 relation')
tf.app.flags.DEFINE_integer("train_sample_num",195635677,"train sample num")
tf.app.flags.DEFINE_integer("dev_sample_num",1327682,"dev sample num")
tf.app.flags.DEFINE_boolean("do_train",True,"whether to run training")
tf.app.flags.DEFINE_boolean("do_eval",False,"whether to run eval on the dev")
tf.app.flags.DEFINE_boolean("do_predict",False,"whether to run model in inference")
tf.app.flags.DEFINE_string("output_dir","./output_v4_geometric_loss_mask/checkpoint",'save model path')
tf.app.flags.DEFINE_integer("save_checkpoints_steps",5000, "how many steps to make estimator call")

#train_sample_file = list()
#rootdir = './data/tfrecord'
#files = os.listdir(rootdir)
#for i in range(0, len(files)):
#    path = os.path.join(rootdir, files[i])
#    train_sample_file.append(path)

#command = "ls data/tfrecord/ | grep part | awk '{print $NF}' 2>/dev/null"
#train_sample_file = FLAGS.train_sample_file

command = "hadoop fs -ls hdfs://ns1013/user/recsys/suggest/app.db/qp_common_file/query_tag/tfrecord/mutitask/dim=v2/dt=2020-12-28/train/ | grep part | awk '{print $NF}' 2>/dev/null"
train_sample_file = run_shell_cmd(command)


def get_relation(filename, flag=False):
    data = np.loadtxt(filename, dtype=np.float32)
    if flag:
        return data
    return data.T

def model_fn_builder(vocab_size, cid2_weight, cid3_weight, brand_weight, relation_matrix, product_matrix, brand_matrix):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(" name = %s, shape = %s" % (name, features[name].shape))
        input_x = features["term_idx"]
        input_y_first = features["cid2_idx"]
        input_y_second = features["cid3_idx"]
        input_product = features["pid_idx"]
        input_brand = features["bid_idx"]
        input_tag = features["tag_idx"]
        model = HierarchicalAttention( FLAGS.keep_prob, cid2_weight, cid3_weight, brand_weight, relation_matrix, product_matrix, brand_matrix, input_x, input_y_first, input_y_second, input_product, input_brand, 
                                       input_tag, FLAGS.fc_size, FLAGS.attention_unit_size, FLAGS.alpha, FLAGS.threshold,
                                       FLAGS.num_classes_first, FLAGS.num_classes_second, FLAGS.num_classes_third,
                                       FLAGS.num_classes_product+1, FLAGS.num_classes_brand+1, FLAGS.num_tags, FLAGS.learning_rate, 
                                       FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length, 
                                       vocab_size, FLAGS.embed_size, FLAGS.hidden_size, FLAGS.is_training)

        total_loss, cid_f1, cid_recall, cid_precision = model.loss_val, model.cid_f1, model.cid_recall, model.cid_precision
        #product_f1, product_recall, product_precision = model.product_f1, model.product_recall, model.product_precision
        #brand_f1, brand_recall, brand_precision = model.brand_f1, model.brand_recall, model.brand_precision
        tag_acc = model.tag_acc
        cid2_loss, cid3_loss, brand_loss, product_loss, tag_loss = model.loss_first, model.loss_second, model.loss_brand, model.loss_product, model.loss_tag
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = model.train_op
            global_step = tf.train.get_or_create_global_step()
            logged_tensors = {
                "global_step": global_step,
                "total_loss": total_loss,
                "cid2_loss": cid2_loss,
                "cid3_loss": cid3_loss,
                "brand_loss": brand_loss,
                "product_loss": product_loss,
                "tag_loss": tag_loss,
                "cid_precision": cid_precision,
                "cid_recall": cid_recall,
                "cid_f1": cid_f1,
                "tag_acc": tag_acc,
                }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=None,
                    training_hooks=[tf.train.LoggingTensorHook(logged_tensors, every_n_iter=2000)])
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(tags):
                return {
                        "eval_loss": total_loss,
                        "precision": cid_precision,
                        "recall": cid_recall,
                        "f1": cid_f1
                        }
                eval_metrics = (metric_fn, [tags])
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode,
                        loss=total_loss,
                        eval_metrics=eval_metrics,
                        scaffold_fn=None)
        else:
            output_sec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={"predictions": model.predictions},
                                  scaffold_fn=None)
        return output_spec
    return model_fn


def file_based_input_fn_builder(num_cpu_threads, input_file, batch_size, seq_length, first_length, second_length, third_length, is_training, drop_remainder):
    #y_length = first_length + second_length + third_length
    name_to_features = {
        'term_idx': tf.FixedLenFeature([seq_length], tf.int64),
        'tag_idx': tf.FixedLenFeature([seq_length], tf.int64),
        'cid2_idx': tf.FixedLenFeature([first_length], tf.int64),
        'cid3_idx': tf.FixedLenFeature([second_length], tf.int64),
        'bid_idx': tf.FixedLenFeature([second_length], tf.int64),
        'pid_idx': tf.FixedLenFeature([second_length], tf.int64),
    }

    def pad_or_trunc(t):
        k = seq_length
        dim = tf.size(t)
        return tf.cond(tf.equal(dim, k), lambda:t, lambda: tf.cond(tf.greater(dim, k), lambda: tf.slice(t, 0, k), lambda: tf.concat([t, tf.zeros(k-dim, dtype=tf.int32)], 0)))


    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        #for name in list(example.keys()):
        #    t = example[name]
        #    if t.dtype == tf.int64:
        #        t = tf.to_int32(t)
        #    example[name] = t

        #example["input_x"] = pad_or_trunc(example["input_x"])
        one_hot_enc = tf.one_hot(indices=example["cid2_idx"], depth=FLAGS.num_classes_first)
        example["cid2_idx"] = tf.reduce_sum(one_hot_enc, axis=0)
        one_hot_enc = tf.one_hot(indices=example["cid3_idx"], depth=FLAGS.num_classes_second)
        example["cid3_idx"] = tf.reduce_sum(one_hot_enc, axis=0)


        #example["pid_idx"] = [example["pid_idx"][0]]
        #example["bid_idx"] = [example["bid_idx"][0]]
        #one_hot_enc = tf.one_hot(indices=example["pid_idx"], depth=FLAGS.num_classes_product)
        #example["pid_idx"] = tf.reduce_sum(one_hot_enc, axis=0)
        #one_hot_enc = tf.one_hot(indices=example["bid_idx"], depth=FLAGS.num_classes_brand)
        #example["bid_idx"] = tf.reduce_sum(one_hot_enc, axis=0)
        
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
    
    #1. load vocabulary
    vocabulary_word2index, vocabulary_index2word= create_term(FLAGS.term_index_path)
    vocab_size = len(vocabulary_index2word) + 2
    print("vocab_size:",vocab_size)
   
    #2. init weight and relation matrix 
    cid2_weight, cid3_weight, brand_weight, relation_matrix, product_matrix, brand_matrix = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    if FLAGS.relation_flag:
        relation_matrix = get_relation(FLAGS.cid2_cid3_relation, flag=True)
        #product_matrix = get_relation(FLAGS.cid3_product_relation)
        #brand_matrix = get_relation(FLAGS.cid3_brand_relation)
    if FLAGS.weight_flag:
        cid2_weight = get_relation(FLAGS.cid2_weight)
        cid3_weight = get_relation(FLAGS.cid3_weight)
        brand_weight = get_relation(FLAGS.brand_weight)
    model_fn = model_fn_builder(
            vocab_size=vocab_size, cid2_weight=cid2_weight, cid3_weight=cid3_weight, brand_weight=brand_weight, relation_matrix=relation_matrix, product_matrix=product_matrix, brand_matrix=brand_matrix)
    
    estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=None,
                model_fn=model_fn,
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
                        seq_length=FLAGS.sequence_length,
                        first_length=FLAGS.first_length,
                        second_length=FLAGS.second_length,
                        third_length=FLAGS.third_length,
                        is_training=True,
                        drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

if __name__ == "__main__":
    tf.app.run()
