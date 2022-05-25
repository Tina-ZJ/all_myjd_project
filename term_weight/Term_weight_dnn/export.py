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



from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.estimator.export.export_output import PredictOutput
SIGNATURE_NAME='serving_default'


#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_cpu_threads",30,"number of cpu")
tf.app.flags.DEFINE_float("keep_prob",0.5,"keep_prob")  
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")    
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.app.flags.DEFINE_integer("vocab_size", 100000, "vocab size")
tf.app.flags.DEFINE_integer("decay_steps", 32000, "how many steps before decay learning rate.")  #24000
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("margin", 0.5, "triple loss margin.")
tf.app.flags.DEFINE_integer("query_len",8,"max query length")   
tf.app.flags.DEFINE_integer("item_len",20,"max item length")   
tf.app.flags.DEFINE_integer("term_dim",200,"term embedding size")  
tf.app.flags.DEFINE_integer("hidden_size",256,"hidden size")
tf.app.flags.DEFINE_integer("num_epochs",3,"number of epochs to run.")  # 6 10 12 8 3 
tf.app.flags.DEFINE_string("term_path","data/term_index.txt",'path of term')
tf.app.flags.DEFINE_integer("train_sample_num",195635677,"train sample num")
tf.app.flags.DEFINE_integer("dev_sample_num",1327682,"dev sample num")
tf.app.flags.DEFINE_boolean("do_train",True,"whether to run training")
tf.app.flags.DEFINE_boolean("do_eval",False,"whether to run eval on the dev")
tf.app.flags.DEFINE_boolean("do_predict",False,"whether to run model in inference")
tf.app.flags.DEFINE_boolean("sub_flag",False,"whether to run model in inference")
tf.app.flags.DEFINE_boolean("use_onehot_emb",False,"whether to use one hot embedding to extract")
tf.app.flags.DEFINE_boolean("is_training",False,"train flag")
tf.app.flags.DEFINE_string("output_dir","./output/checkpoint",'save model path')
tf.app.flags.DEFINE_string("init_checkpoint",None,'init  model parameter from pretrain model')
tf.app.flags.DEFINE_integer("save_checkpoints_steps",10000, "how many steps to make estimator call")



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
        input_query = features["input_query"]
        pos_item = features["pos_item"]
        #neg_item = features["neg_item"]
        neg_item = pos_item

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
       
   
        # show result 
        if mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={"query_weight": model.query_weight,
                                 "pos_weight": model.pos_weight,
                                 "pos_similar": model.pos_similar})
        return output_spec
    return model_fn


def serving_input_receiver_fn():
    input_query = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_query')
    pos_item = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos_item')
    receiver_tensors = {'input_query': input_query,
                        'pos_item': pos_item}

    features = {'input_query': input_query,
                'pos_item': pos_item}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors) 
     
            
        
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
    term_size = max(idx2word.keys()) + 1
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

    cp_file = tf.train.latest_checkpoint(FLAGS.output_dir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.log_device_placement = False
    batch_size = 1
    export_dir = FLAGS.output_dir
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.output_dir, config=RunConfig(session_config=config),
                                       params={'batch_size': batch_size})
    estimator.export_saved_model(export_dir, serving_input_receiver_fn, checkpoint_path=cp_file, as_text=True) 


if __name__ == "__main__":
    tf.app.run()
