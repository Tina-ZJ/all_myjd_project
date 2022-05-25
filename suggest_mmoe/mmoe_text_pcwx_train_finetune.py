#!/usr/bin/env python3
# coding=utf-8
import os
import json

os.environ['TF_CONFIG'] = json.dumps({})
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import horovod.tensorflow as hvd
from base_func import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("epochs", 50, "epochs")
tf.app.flags.DEFINE_integer("batch_size", 512, "batch size")
tf.app.flags.DEFINE_integer("steps_epoch_num", 1000, "steps_epoch_num")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning_rate")
tf.app.flags.DEFINE_string("model_version", "default_version", "model version")

tf.app.flags.DEFINE_string("checkpoint_path", "./checkpoint", "model path")
tf.app.flags.DEFINE_string("train_file_path",
                           "hdfs://ns1013/user/recsys/suggest/app.db/sample_realt/V4024_fix/v3/20210309/week/traintfrecord/*",
                           "train file path")
tf.app.flags.DEFINE_string("valid_file_path",
                           "hdfs://ns1013/user/recsys/suggest/app.db/sample_realt/V4024/v3/20210309/valtfrecord/*",
                           "valid file path")


def _parse_batch_function(example_proto):
    context_features = {
        "channel": tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64, allow_missing=True, default_value=0),
        "sug_stac_zscore_fea": tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True,
                                                       default_value=0.0),
        "sug_stac_max_fea": tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True,
                                                      default_value=0.0),
        "pref_stac_zscore_fea": tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True,
                                                             default_value=0.0),
        "pref_stac_max_fea": tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True,
                                                          default_value=0.0),
        "prefsug_stac_fea": tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True,
                                                          default_value=0.0),
        "prefix_word": tf.FixedLenFeature(shape=(1,), dtype=tf.string),
        "suggest_word": tf.FixedLenFeature(shape=(1,), dtype=tf.string), 
        "click_label": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
        "order_label": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64)
    }

    context = tf.io.parse_example(serialized=example_proto, features=context_features)

    return ({"channel": context["channel"],
             "sug_stac_zscore_fea": context["sug_stac_zscore_fea"],
             "sug_stac_max_fea": context["sug_stac_max_fea"],
             "pref_stac_zscore_fea": context["pref_stac_zscore_fea"],
             "pref_stac_max_fea": context["pref_stac_max_fea"],
             "prefsug_stac_fea": context["prefsug_stac_fea"],
             "prefix": context["prefix_word"],
             "suggest": context["suggest_word"]},
            {"click_label": context["click_label"],
             "order_label": context["order_label"]})


def binary_PFA(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, 'float32')
    N = tf.reduce_sum(1 - y_true)
    FP = tf.reduce_sum(y_pred - y_pred * y_true)
    return FP / N


def binary_PTA(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, 'float32')
    P = tf.reduce_sum(y_true)
    TP = tf.reduce_sum(y_pred * y_true)
    return TP / P * 1.0


def auc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return tf.reduce_sum(s, axis=0)


def main(_):
    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    epochs = FLAGS.epochs
    steps_epoch_num = 3000
    checkpoint_path = FLAGS.checkpoint_path
    train_file_path = FLAGS.train_file_path
    valid_file_path = FLAGS.valid_file_path
    model_version = FLAGS.model_version
    app_model_path = None
    app_checkpoint_path = None
    # app checkpoint path
    app_check_path = checkpoint_path.replace('VPCWX2022', 'VAPP2022')
    if os.path.exists(app_check_path):
        app_checkpoint_path = app_check_path
        app_model_path = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=app_checkpoint_path)
    # control early stop
    max_steps_without_increase=20000
    min_steps=80000
     
    print('======checkpoint_path==========:', checkpoint_path)
    print('======app_checkpoint_path==========:', app_checkpoint_path)
    print('======train_file_path=====:', train_file_path)
    print('======valid_file_path=====:', valid_file_path)

    hvd.init()

    def input_data_fn(batch_size, mode):
        def input_fn():
            if mode == 'train':
                train_dataset = get_dataset(train_file_path, batch_size, _parse_batch_function)
                train_dataset = train_dataset.shard(hvd.size(), hvd.rank())
                return train_dataset
            if mode == 'eval':
                val_dataset = get_dataset(valid_file_path, batch_size, _parse_batch_function)
                val_dataset = val_dataset.shard(hvd.size(), hvd.rank())
                return val_dataset

        return input_fn

    def xiala_model_fn(features, labels, mode):

#         def dnn_net(emb_feature, static_feature):
#             dnn_l1 = tf.layers.dense(static_feature, 64, kernel_initializer=tf.initializers.he_uniform())
#             dnn_l1 = tf.nn.selu(dnn_l1)
#             dnn_l2 = tf.layers.dense(emb_feature, 64, kernel_initializer=tf.initializers.he_uniform())
#             dnn_l2 = tf.nn.selu(dnn_l2)
#             multi_l1_l2 = tf.multiply(dnn_l1, dnn_l2)
#             dnn_l3 = tf.layers.dense(multi_l1_l2, 64, kernel_initializer=tf.initializers.he_uniform())
#             dnn_l3 = tf.nn.selu(dnn_l3)
#             return dnn_l3
        
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
 
        batch = tf.shape(features['sug_stac_zscore_fea'])[0]

        sug_stac_zscore_fea = tf.reshape(features['sug_stac_zscore_fea'], [batch, 39])
        sug_stac_max_fea = tf.reshape(features['sug_stac_max_fea'], [batch, 39])
        pref_stac_zscore_fea = tf.reshape(features['pref_stac_zscore_fea'], [batch, 51])
        pref_stac_max_fea = tf.reshape(features['pref_stac_max_fea'], [batch, 51])
        prefsug_stac_fea = tf.reshape(features['prefsug_stac_fea'], [batch, 2])
        #text_fea = tf.reshape(features['text_fea'], [batch, 5])
        text_fea = tf.reshape(text_func(features['prefix'],features['suggest']), [batch, 5])
        channel = tf.reshape(features['channel'], [batch, 1])
        
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
#         static_zscore_fea = tf.concat([sug_stac_zscore_fea, pref_stac_zscore_fea, prefsug_stac_fea, text_fea, channel_emb], axis=-1)
#         static_max_fea = tf.concat([sug_stac_max_fea, pref_stac_max_fea, prefsug_stac_fea, text_fea, channel_emb], axis=-1)
#         full_feature = tf.concat([sug_stac_zscore_fea, sug_stac_max_fea, pref_stac_zscore_fea, pref_stac_max_fea,
#                                   prefsug_stac_fea, text_fea, channel_emb_gate], axis=-1)
        
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
        
        app_weight = tf.fill([batch, 1], 1.0)
        pcwx_weight = tf.fill([batch, 1], 2.0)
        
        if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
            # Calculate Loss (for both TRAIN and EVAL modes)
            click_labels = tf.cast(labels['click_label'], tf.float32, name='click_labels')
            order_labels = tf.cast(labels['order_label'], tf.float32, name='order_labels')
            click_classes = tf.cast(tf.greater(tf.nn.sigmoid(ctr_out), 0.5), tf.float32)
            order_classes = tf.cast(tf.greater(tf.nn.sigmoid(cvr_out), 0.5), tf.float32)
            loss_weight = tf.where(tf.greater(channel, 0), pcwx_weight, app_weight)
            click_loss = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=click_labels, logits=ctr_out), loss_weight),
                                        name='click_loss')
            order_loss = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=order_labels, logits=cvr_out), loss_weight),
                                        name='order_loss')
#             click_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=click_labels, logits=ctr_out),
#                                         name='click_loss')
#             order_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=order_labels, logits=cvr_out),
#                                         name='order_loss')
        
            global_step = tf.train.get_or_create_global_step()
            click_acc = tf.reduce_mean(tf.cast(tf.equal(click_classes, click_labels), tf.float32), name='click_acc')
            order_acc = tf.reduce_mean(tf.cast(tf.equal(order_classes, order_labels), tf.float32), name='order_acc')
            total_loss = click_loss + order_loss
            
            tf.summary.scalar('click_loss', click_loss)
            tf.summary.scalar('click_acc', click_acc)
            tf.summary.scalar('order_loss', order_loss)
            tf.summary.scalar('order_acc', order_acc)
            tf.summary.scalar('total_loss', total_loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            eval_metric_ops = None
            decay_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                             global_step=global_step,
                                                             decay_steps=steps_epoch_num,
                                                             decay_rate=0.96,
                                                             staircase=True,
                                                             name='learning_rate')

            optimizer = tf.train.AdamOptimizer(decay_learning_rate)
            optimizer = hvd.DistributedOptimizer(optimizer)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # 保证train_op在update_ops执行之后再执行。
            # g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator_net')
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss=total_loss, global_step=global_step)

            tf.summary.scalar('learning_rate', decay_learning_rate)
            saver = tf.train.Saver(sharded=True, max_to_keep=2, var_list=tf.global_variables())
            tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
            scaffold = tf.train.Scaffold(saver=saver)

        # Add evaluation metrics (for EVAL mode)
        if mode == tf.estimator.ModeKeys.EVAL:
            train_op = None
            scaffold = None
            click_auc = tf.metrics.auc(labels=click_labels, predictions=tf.nn.sigmoid(ctr_out), name='eval_click_auc')
            click_acc = tf.metrics.accuracy(labels=click_labels, predictions=click_classes)
            order_auc = tf.metrics.auc(labels=order_labels, predictions=tf.nn.sigmoid(cvr_out), name='eval_order_auc')
            order_acc = tf.metrics.accuracy(labels=order_labels, predictions=order_classes)
            eval_metric_ops = {
                'eval_click_auc': click_auc,
                'eval_click_acc': click_acc,
                'eval_order_auc': order_auc,
                'eval_order_acc': order_acc,
            }

        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     export_output = exporter._add_output_tensor_nodes(predictions)
        #     export_outputs = {
        #           tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
        #           tf.estimator.export.PredictOutput(export_output)
        #         }
        #
        summary_hook = tf.estimator.SummarySaverHook(save_steps=500, output_dir=model_dir,
                                                     summary_op=tf.summary.merge_all())
        hooks = [summary_hook]
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions={'ctr_out': ctr_out, 'cvr_out': cvr_out},
                                          loss=total_loss,
                                          train_op=train_op,
                                          training_chief_hooks=hooks,
                                          eval_metric_ops=eval_metric_ops,
                                          scaffold=scaffold)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    # cpu_num = 60
    # config = tf.ConfigProto(device_count={"CPU": cpu_num},
    #                         inter_op_parallelism_threads=cpu_num,
    #                         intra_op_parallelism_threads=cpu_num,
    #                         log_device_placement=True)

    model_dir = checkpoint_path if hvd.rank() == 0 else None

    model_estimator = tf.estimator.Estimator(
        model_fn=xiala_model_fn,
        model_dir=model_dir,
        warm_start_from=app_model_path,
        config=tf.estimator.RunConfig(save_checkpoints_steps=2000,
                                      keep_checkpoint_max=2,
                                      save_summary_steps=500,
                                      log_step_count_steps=500,
                                      session_config=config))

    tensors_to_log = {'learning_rate': 'learning_rate', 'click_acc': 'click_acc', 'order_acc': 'order_acc'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)
    early_stopping = tf.estimator.experimental.stop_if_no_increase_hook(model_estimator,
                                                                        metric_name='eval_click_auc',
                                                                        max_steps_without_increase=max_steps_without_increase,
                                                                        min_steps=min_steps,
                                                                        run_every_steps=2000,
                                                                        run_every_secs=None)
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_data_fn(batch_size, mode='train'),
        max_steps=epochs * steps_epoch_num // hvd.size(),
        hooks=[hvd.BroadcastGlobalVariablesHook(0), logging_hook, early_stopping])

    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_data_fn(batch_size, mode='eval'),
        throttle_secs=30,
        steps=250,
        start_delay_secs=10,
        hooks=[hvd.BroadcastGlobalVariablesHook(0)])

    print('start training...')

    tf.estimator.train_and_evaluate(model_estimator, train_spec, eval_spec)

    print('end training!')


if __name__ == '__main__':
    tf.app.run()
