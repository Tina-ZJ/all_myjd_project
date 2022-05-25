#!/usr/bin/env python3
# coding=utf-8
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("model_version", "default_version", "model version")
tf.app.flags.DEFINE_string("model_path", "./checkpoint", "model path")
tf.app.flags.DEFINE_string("save_pb_path", "./pb_model", "model path")


def xiala_model_fn(features, mode):
    def dnn_net(input_feature):
        dnn_l1 = tf.layers.dense(input_feature, 128, kernel_initializer=tf.initializers.he_uniform())
        dnn_l1 = tf.nn.selu(dnn_l1)
        dnn_l2 = tf.layers.dense(dnn_l1, 64, kernel_initializer=tf.initializers.he_uniform())
        dnn_l2 = tf.nn.selu(dnn_l2)
        return dnn_l2

    def tower_net(input_feature):
        dnn_l1 = tf.layers.dense(input_feature, 32, kernel_initializer=tf.initializers.he_uniform())
        dnn_l1 = tf.nn.selu(dnn_l1)
        return dnn_l1

    def gate_net(input_feature):
        dnn_l1 = tf.layers.dense(input_feature, 2, kernel_initializer=tf.initializers.he_uniform())
        dnn_l1 = tf.nn.softmax(dnn_l1)
        return dnn_l1
    
    def text_func(prefix, suggest):
        pref_len = tf.strings.length(prefix, unit='UTF8_CHAR')
        sug_len = tf.strings.length(suggest, unit='UTF8_CHAR')
        pref_v = tf.reshape(prefix, [-1])
        sug_v = tf.reshape(suggest, [-1])
        sp_h = tf.strings.unicode_split(pref_v, 'UTF-8').to_sparse()
        sp_t = tf.strings.unicode_split(sug_v, 'UTF-8').to_sparse()
        lev = tf.edit_distance(sp_h, sp_t, normalize=False)
        lev = tf.reshape(lev, [-1, 1])
        text_fea_list = tf.concat(
            [tf.cast(lev, tf.float32),
            tf.cast(pref_len, tf.float32),
            tf.cast(sug_len, tf.float32),
            tf.cast((1 + pref_len) / (1 + sug_len), tf.float32),
            tf.cast(tf.abs((1 + pref_len) - (1 + sug_len)), tf.float32)
            ], axis=-1)
        return text_fea_list

    sug_stac_zscore_fea = features['sug_stac_zscore_fea']
    sug_stac_max_fea = features['sug_stac_max_fea']
    pref_stac_zscore_fea = features['pref_stac_zscore_fea']
    pref_stac_max_fea = features['pref_stac_max_fea']
    prefsug_stac_fea = features['prefsug_stac_fea']
    text_fea = text_func(features['prefix'],features['suggest'])

    channelEmb = tf.get_variable(name='channelEmb', shape=[3, 50],
                                initializer=tf.random_uniform_initializer(), trainable=True)
    
    channelEmbGate = tf.get_variable(name='channelEmbGate', shape=[3, 50],
                                initializer=tf.random_uniform_initializer(), trainable=True)
    # mask 部分异常值
    channel_realt_mask = tf.cast(tf.less_equal(features['channel'], 2), tf.int32)
    channel_realt_mask = tf.multiply(tf.cast(features['channel'], tf.int32), channel_realt_mask)
    channel_realt_min = tf.cast(tf.greater_equal(channel_realt_mask, 0), tf.int32)
    channel_realt_mask = tf.multiply(channel_realt_mask, channel_realt_min)

    channel_emb = tf.nn.embedding_lookup(channelEmb, channel_realt_mask)
    channel_emb = tf.reduce_sum(channel_emb, 1)
    
    channel_emb_gate = tf.nn.embedding_lookup(channelEmbGate, channel_realt_mask)
    channel_emb_gate = tf.reduce_sum(channel_emb_gate, 1)
    # ############
    static_zscore_fea = tf.concat([sug_stac_zscore_fea, pref_stac_zscore_fea, prefsug_stac_fea, text_fea], axis=-1)
    static_max_fea = tf.concat([sug_stac_max_fea, pref_stac_max_fea, prefsug_stac_fea, text_fea], axis=-1)
    full_feature = tf.concat([sug_stac_zscore_fea, sug_stac_max_fea, pref_stac_zscore_fea, pref_stac_max_fea,
                              prefsug_stac_fea, text_fea], axis=-1)

    # static-feature expert-net
    static_zscore_expert = dnn_net(static_zscore_fea)
    static_max_expert = dnn_net(static_max_fea)

    # gate-net
    ctr_gate_out = gate_net(channel_emb)
    cvr_gate_out = gate_net(channel_emb)

    # mmoe-cross
    expert_out = tf.concat([static_zscore_expert, static_max_expert], axis=-1)
    mmoe_ctr_cross = expert_out * tf.repeat(ctr_gate_out, [64, 64], axis=1)
    mmoe_cvr_cross = expert_out * tf.repeat(cvr_gate_out, [64, 64], axis=1)

    # mmoe-weighted-sum
    mmoe_ctr_out = tf.reduce_sum(tf.reshape(mmoe_ctr_cross, [-1, 2, 64]), axis=1)
    mmoe_cvr_out = tf.reduce_sum(tf.reshape(mmoe_cvr_cross, [-1, 2, 64]), axis=1)

    # tower-output
    ctr_tower = tower_net(mmoe_ctr_out)
    cvr_tower = tower_net(mmoe_cvr_out)

    ctr_out = tf.layers.dense(ctr_tower, 1, kernel_initializer=tf.glorot_uniform_initializer(), name='click_pred')
    cvr_out = tf.layers.dense(cvr_tower, 1, kernel_initializer=tf.glorot_uniform_initializer(), name='order_pred')

    output = tf.identity((tf.nn.sigmoid(ctr_out) + tf.nn.sigmoid(cvr_out))/2.0, name='pred/Sigmoid')
    return output


def export_pb_model(checkpoint_path, pb_path):
    with tf.Graph().as_default():
        channel = tf.placeholder(tf.int32, shape=(None, 1), name='channel')
        sug_stac_zscore_fea = tf.placeholder(tf.float32, shape=(None, 39), name='sug_stac_zscore_fea')
        sug_stac_max_fea = tf.placeholder(tf.float32, shape=(None, 39), name='sug_stac_max_fea')
        pref_stac_zscore_fea = tf.placeholder(tf.float32, shape=(None, 51), name='pref_stac_zscore_fea')
        pref_stac_max_fea = tf.placeholder(tf.float32, shape=(None, 51), name='pref_stac_max_fea')
        prefsug_stac_fea = tf.placeholder(tf.float32, shape=(None, 2), name='prefsug_stac_fea')
        prefix = tf.placeholder(tf.string, shape=(None, 1), name='prefix')
        suggest = tf.placeholder(tf.string, shape=(None, 1), name='suggest')
        #text_fea = tf.placeholder(tf.float32, shape=(None, ), name='text_fea')
        input_tensor = {
            'channel': channel,
            'sug_stac_zscore_fea': sug_stac_zscore_fea,
            'sug_stac_max_fea': sug_stac_max_fea,
            'pref_stac_zscore_fea': pref_stac_zscore_fea,
            'pref_stac_max_fea': pref_stac_max_fea,
            'prefsug_stac_fea': prefsug_stac_fea,
            'prefix': prefix,
            'suggest': suggest
        }
        output = xiala_model_fn(input_tensor, mode="predict")

        model_signature = signature_def_utils.build_signature_def(
            inputs={
                'channel': utils.build_tensor_info(channel),
                'sug_stac_zscore_fea': utils.build_tensor_info(sug_stac_zscore_fea),
                'sug_stac_max_fea': utils.build_tensor_info(sug_stac_max_fea),
                'pref_stac_zscore_fea': utils.build_tensor_info(pref_stac_zscore_fea),
                'pref_stac_max_fea': utils.build_tensor_info(pref_stac_max_fea),
                'prefsug_stac_fea': utils.build_tensor_info(prefsug_stac_fea),
                'prefix': utils.build_tensor_info(prefix),
                'suggest': utils.build_tensor_info(suggest)
            },
            outputs={"pred": utils.build_tensor_info(output)},
            method_name=signature_constants.PREDICT_METHOD_NAME)

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )

        sess = tf.Session(config=session_conf)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        export_path = compat.as_bytes(pb_path)
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={'serving_default': model_signature})

        builder.save(as_text=True)
        print("save model pb success ...")


if __name__ == '__main__':
    model_path = FLAGS.model_path
    save_pb_path = FLAGS.save_pb_path
    model_version = FLAGS.model_version
    print("model_path")
    print(model_path)
    print("model_version")
    print(model_version)
    print("save_pb_path")
    print(save_pb_path)
    export_pb_model(checkpoint_path=model_path, pb_path=save_pb_path)
