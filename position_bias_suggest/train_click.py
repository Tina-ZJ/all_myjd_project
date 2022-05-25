# -*- coding: utf-8 -*-
#import sys
import tensorflow as tf
import re
import numpy as np
import modeling
import collections
import tensorflow.contrib as tf_contrib
#from data_util import create_term
import os
import codecs
import traceback
from read_file import run_shell_cmd

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_cpu_threads",30,"number of cpu")
tf.app.flags.DEFINE_float("drop_prob",0.5,"drop_prob")  # 0.2
tf.app.flags.DEFINE_float("lr",0.001,"learning rate")  #0.001 0.01  
tf.app.flags.DEFINE_float("sigma",1.,"sigma")  
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 32000, "how many steps before decay learning rate.")  #24000
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_integer("seq_len",55,"max sentence length")   # 47  55
tf.app.flags.DEFINE_integer("term_dim",200,"term embedding size")  
tf.app.flags.DEFINE_integer("gender_size",4,"gender nums")  
tf.app.flags.DEFINE_integer("gender_dim",64,"gender embedding size")  
tf.app.flags.DEFINE_integer("age_size",6,"age nums")  
tf.app.flags.DEFINE_integer("age_dim",64,"age embedding size")  
tf.app.flags.DEFINE_integer("cidx_dim",128,"cidx embedding size")  
tf.app.flags.DEFINE_integer("hidden1_size",512,"hidden1 size")
tf.app.flags.DEFINE_integer("hidden2_size",256,"hidden2 size")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")  # 6 10 12 8 3 
tf.app.flags.DEFINE_string("term_path","data/term_index_v2.txt",'path of term')
tf.app.flags.DEFINE_string("cidx_path","data/cidx_index_8_25.txt",'path of cid')
tf.app.flags.DEFINE_integer("train_sample_num",195635677,"train sample num")
tf.app.flags.DEFINE_integer("dev_sample_num",1327682,"dev sample num")
tf.app.flags.DEFINE_boolean("do_train",True,"whether to run training")
tf.app.flags.DEFINE_boolean("do_eval",False,"whether to run eval on the dev")
tf.app.flags.DEFINE_boolean("do_predict",False,"whether to run model in inference")
tf.app.flags.DEFINE_boolean("sub_flag",False,"whether to run model in inference")
tf.app.flags.DEFINE_boolean("use_onehot_emb",False,"whether to use one hot embedding to extract")
tf.app.flags.DEFINE_string("output_dir","./output_click_one_v2/checkpoint",'save model path')
tf.app.flags.DEFINE_string("init_checkpoint",None,'init  model parameter from pretrain model')
tf.app.flags.DEFINE_string("bert_config_file",'./bert_config_large.json', 'bert config file') #bert_config_v2.json
tf.app.flags.DEFINE_integer("save_checkpoints_steps",2000, "how many steps to make estimator call")


#command = "hadoop fs -ls hdfs://ns1013/user/recsys/suggest/app.db/qp_common_file/suggest_rank/2022-03-11/bert/v5/ | grep part | awk '{print $NF}' 2>/dev/null"
command = "hadoop fs -ls hdfs://ns1013/user/recsys/suggest/app.db/qp_common_file/suggest_rank/2022-03-12/bert/label/position/ | grep part | awk '{print $NF}' 2>/dev/null"


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
        


def inference(model):
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    
    #projection
    score = tf.layers.dense(output_layer,1 , activation=None)
    #predictions = tf.sigmoid(logits, name='predictions')
    return score
     

def jacobian_(score, Wk):
    
    result = list()
    for s in score:
        g = tf.gradients(s, Wk)
        result.append(g) 

    return result.stack()

def jacobian(score, Wk):
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=FLAGS.batch_size),
    ]
    _, jacobian = tf.while_loop(
        lambda j, _: j < FLAGS.batch_size,
        lambda j, result: (j + 1, result.write(j, tf.gradients(score[j], Wk))), loop_vars)

    return jacobian.stack()


def get_derivative(score, Wk, lambda_ij):
    
    dsi_dwk = jacobian(score, Wk)
    dsi_dwk_minus_dsj_dwk = tf.expand_dims(dsi_dwk, 1) - tf.expand_dims(dsi_dwk, 0)
    shape = tf.concat(
        [tf.shape(lambda_ij), tf.ones([tf.rank(dsi_dwk_minus_dsj_dwk) - tf.rank(lambda_ij)], dtype=tf.int32)],
        axis=0)
    grad = tf.reduce_mean(tf.reshape(lambda_ij, shape) * dsi_dwk_minus_dsj_dwk, axis=[0,1])
    return tf.reshape(grad, tf.shape(Wk))



def lambda_rank(score, labels, position, click_labels, group_id, sigma, lr):

    S_ij = labels - tf.transpose(labels)
    S_ij = tf.maximum(tf.minimum(1.0, tf.cast(S_ij, tf.float32)), -1.)
    #S_ij = -S_ij
    P_ij = (1.0 /2.0) * (1 + S_ij)
    s_i_minus_s_j = logits = score - tf.transpose(score)
    #lambda_ij = sigma * ((1.0/2.0)*(1 - S_ij) - tf.nn.sigmoid(sigma*s_i_minus_s_j))
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=s_i_minus_s_j, labels=P_ij)
    
    # only extracted the sample group id pair loss
    mask1 = tf.equal(group_id - tf.transpose(group_id), 0)
    mask1 = tf.cast(mask1, tf.float32)
    # exclude itself
    n = tf.shape(score)[0]
    mask2 = tf.ones([n,n]) - tf.diag(tf.ones([n]))
    # combine
    mask = mask1 * mask2
    #num_pairs = tf.reduce_sum(mask)
    # masked loss
    #loss = loss * mask


    # position v3 add
    mask_p = tf.cast(tf.less(position, 14), tf.int64)
    position = position * mask_p

    # position
    position_mat = tf.cast(position - tf.transpose(position), dtype=tf.float32)
    #position_mask = tf.equal(position_mat,0)

    #pad 0 with 1
    #position_1 = tf.cast(position_mat, tf.float32) + tf.cast(position_mask,tf.float32)
    position_1 = 1.0 + (1.0/16.0) * position_mat
    
    #position_2 = -1/position_1
    position_2 = 1 - (1.0/80.0)*(-position_mat)

    #position_3 = 1/position_1
    position_3 = 1.0 - (1.0/80.0)*position_mat
    
    #position_4 = -position_1
    position_4 = 1.0 + (1.0/16.0)*(-position_mat)
 
    # get weight matrix
    weight_pos = tf.where(P_ij >0., position_1, position_3) 
    weight_neg = tf.where(P_ij >0., position_2, position_4)
    # position compare 
    mask_pos = tf.cast(tf.greater(position_1,0.), tf.float32)
    mask_neg = 1.0 - mask_pos
 
    # final weight
    weight_pos = weight_pos * mask_pos
    weight_neg = weight_neg * mask_neg
    weight_final = weight_pos + weight_neg

    # only compare click before pairs 
    labels_mat = click_labels - tf.transpose(click_labels)
    labels_mask = tf.maximum(tf.minimum(1.0, tf.cast(labels_mat, tf.float32)), -0.)

    # add global sample
    position_label = 1 - mask_p
    p_mat = position_label - tf.transpose(position_label)
    position_mask = tf.maximum(tf.minimum(1.0, tf.cast(labels_mat, tf.float32)), -0.)
    
    # combine two 
    com_mask = labels_mask + position_mask 

    # weighted loss
    # one
    mask = com_mask * mask
    loss = loss * mask
    num_pairs = tf.reduce_sum(mask)
    # two
    #loss = loss * weight_final 
    # final
 
    loss = tf.cond(tf.equal(num_pairs, 0), lambda: 0., lambda: tf.reduce_sum(loss) / num_pairs)

    #lambda_ij = lambda_ij * mask
    # all vars
    #varis = tf.trainable_variables()
    #grads = [ get_derivative(score, Wk, lambda_ij) for Wk in varis ]
    
    # opimizer
    #optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        #train_op = optimizer.apply_gradients(zip(grads, varis))
        #train_op = optimizer.minimize(loss)
        train_op = train(loss, lr, FLAGS.decay_steps, FLAGS.decay_rate)

    return loss, num_pairs, train_op
 

def multi_loss(logits, labels):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss_mean = tf.reduce_mean(loss, name='loss')
    return loss_mean


def train(loss, lr, decay_steps, decay_rate, clip_gradients=5.0):
    decay_lr = tf.train.exponential_decay(lr, tf.train.get_global_step(), decay_steps, decay_rate, staircase=True)
    train_op = tf_contrib.layers.optimize_loss(loss, global_step=None, learning_rate=decay_lr, optimizer='Adam', clip_gradients=clip_gradients)
    return train_op
 
def model_fn_builder(init_checkpoint, term_size, bert_config):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(" name = %s, shape = %s" % (name, features[name].shape))

        # input features
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        relative_label = features["relative_label"]
        position = features["position"]
        label = features["label"]
        group_id = features["group_id"]
         
        is_training = tf.cast((mode == tf.estimator.ModeKeys.TRAIN), tf.bool, name='is_training')
        model = modeling.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=FLAGS.use_onehot_emb)

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
            score = inference(model)
            
            # lambda rank
            loss, num_pairs, train_op = lambda_rank(score, relative_label, position, label, group_id, FLAGS.sigma, FLAGS.lr ) 
 
            global_step = tf.train.get_or_create_global_step()
            # log 
            logged_tensors = {
            "global_step": global_step,
            "num_pairs": num_pairs,
            "loss": loss,
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


def file_based_input_fn_builder(num_cpu_threads, input_file, batch_size, seq_len, is_training, drop_remainder):
    name_to_features = {
        'input_ids': tf.FixedLenFeature([seq_len], tf.int64),
        'input_mask': tf.FixedLenFeature([seq_len], tf.int64),
        'segment_ids': tf.FixedLenFeature([seq_len], tf.int64),
        'relative_label': tf.FixedLenFeature([1], tf.int64),
        'position': tf.FixedLenFeature([1], tf.int64),
        'label': tf.FixedLenFeature([1], tf.int64),
        'group_id': tf.FixedLenFeature([1], tf.int64),
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

        return example

    def input_fn(params):
        """ The actual input function. """
        if  is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_file))
            d = d.repeat()
            #d = d.shuffle(buffer_size=len(input_file))
            
            cycle_length = min(num_cpu_threads, len(input_file))
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            #d = d.shuffle(buffer_size=25600)
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
    
    #1. load term and cidx
    #word2idx, idx2word= create_term(FLAGS.term_path)
    
    term_size = 634305
    print("term_size:",term_size)
  
    # load bert config
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    model_fn = model_fn_builder(
                init_checkpoint=FLAGS.init_checkpoint, 
                term_size=term_size,
                bert_config=bert_config)
   
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
                        seq_len=FLAGS.seq_len,
                        is_training=True,
                        drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

if __name__ == "__main__":
    tf.app.run()
