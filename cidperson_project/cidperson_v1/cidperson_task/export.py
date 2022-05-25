# -*- coding: utf-8 -*-
#import sys
import tensorflow as tf
import re
import numpy as np
import modeling
import collections
import tensorflow.contrib as tf_contrib
from data_util import create_term
import os
import codecs
import traceback


from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.estimator.export.export_output import PredictOutput
SIGNATURE_NAME='cid_personalization_bert'

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_cpu_threads",30,"number of cpu")
tf.app.flags.DEFINE_float("drop_prob",0.5,"drop_prob")  # 0.2
tf.app.flags.DEFINE_float("lr",0.001,"learning rate")  #0.001 0.01  
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 24000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_integer("seq_len",33,"max sentence length")  #35 28
tf.app.flags.DEFINE_integer("term_dim",200,"term embedding size")  
tf.app.flags.DEFINE_integer("cidx_dim",128,"cidx embedding size")  
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")  # 10 
tf.app.flags.DEFINE_string("term_path","./term_index.txt",'path of term')
tf.app.flags.DEFINE_string("cidx_path","./cidx_index.txt",'path of cid')
tf.app.flags.DEFINE_integer("train_sample_num",195635677,"train sample num")
tf.app.flags.DEFINE_integer("dev_sample_num",1327682,"dev sample num")
tf.app.flags.DEFINE_boolean("do_train",True,"whether to run training")
tf.app.flags.DEFINE_boolean("do_eval",False,"whether to run eval on the dev")
tf.app.flags.DEFINE_boolean("do_predict",False,"whether to run model in inference")
tf.app.flags.DEFINE_boolean("sub_flag",False,"whether to run model in inference")
tf.app.flags.DEFINE_boolean("use_onehot_emb",False,"whether to use one hot embedding to extract")
tf.app.flags.DEFINE_string("output_dir","./checkpoint",'save model path')
tf.app.flags.DEFINE_string("init_checkpoint",None,'init  model parameter from pretrain model')
tf.app.flags.DEFINE_string("bert_config_file",'./bert_config.json', 'bert config file')
tf.app.flags.DEFINE_integer("save_checkpoints_steps",5000, "how many steps to make estimator call")




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
        if name not in name_to_variable :
            continue
        #assignment_map[name] = name
        assignment_map[name] = name_to_variable[name] 
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
    return (assignment_map, initialized_variable_names)
        


def inference(model, cidx_size):
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    
    #projection
    logits = tf.layers.dense(output_layer, cidx_size, activation=None)
    predictions = tf.sigmoid(logits, name='predictions')
    return (logits, predictions)
     

def metric(predictions, labels, batch_size, threshold=0.5):
    
    # precision recall f1
    tp = tf.reduce_sum(tf.cast(tf.greater_equal(predictions, threshold), tf.float32) * tf.cast(tf.equal(labels, 1), tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.less(predictions, threshold), tf.float32) * tf.cast(tf.equal(labels, 0), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.greater_equal(predictions, threshold), tf.float32) * tf.cast(tf.equal(labels, 0), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.less(predictions, threshold), tf.float32) * tf.cast(tf.equal(labels, 1), tf.float32))
    precision = tf.div(tp, tp+fp, name='precision')
    recall = tf.div(tp, tp+fn, name='recall')
    f1 = tf.div(2*precision*recall, precision+recall, name='F1')
    return (precision, recall, f1)


def multi_loss(logits, labels):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss_mean = tf.reduce_mean(loss, name='loss')
    return loss_mean


def train(loss, lr, decay_steps, decay_rate, clip_gradients=5.0):
    decay_lr = tf.train.exponential_decay(lr, tf.train.get_global_step(), decay_steps, decay_rate, staircase=True)
    train_op = tf_contrib.layers.optimize_loss(loss, global_step=None, learning_rate=decay_lr, optimizer='Adam', clip_gradients=clip_gradients)
    return train_op
 
def model_fn_builder(init_checkpoint, term_size, cidx_size, bert_config):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(" name = %s, shape = %s" % (name, features[name].shape))

        # input features
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        #cidx_idx = features["cidx"]
        cidx_idx = tf.ones(tf.shape(input_ids), dtype=tf.int32)
        is_training = tf.cast((mode == tf.estimator.ModeKeys.TRAIN), tf.bool, name='is_training')
        #label = features["label"]
        model = modeling.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=FLAGS.use_onehot_emb)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
       

        logits, predictions = inference(model, cidx_size)   
        # show result 
        if mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={'predictions': predictions},
                    export_outputs={SIGNATURE_NAME: PredictOutput({"predictions": predictions,})})

        return output_spec
    return model_fn


def serving_input_receiver_fn():
     
    input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
    receiver_tensors = {'input_ids': input_ids,
                        'input_mask': input_mask,
                        'segment_ids': segment_ids}
    features = {'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids}
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
    
    #1. load term and cidx
    word2idx, idx2word= create_term(FLAGS.term_path)
    cidx2idx, idx2cidx= create_term(FLAGS.cidx_path)
    term_size = max(idx2word.keys()) + 1 + 12
    cidx_size = max(idx2cidx.keys()) + 1
    print("term_size:",term_size)
    print("cidx_size:",cidx_size)
  
    # load bert config
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    model_fn = model_fn_builder(
                init_checkpoint=FLAGS.init_checkpoint, 
                term_size=term_size, 
                cidx_size=cidx_size,
                bert_config=bert_config)
  
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
