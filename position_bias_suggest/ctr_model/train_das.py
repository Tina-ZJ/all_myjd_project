import das.tensorflow.estimator as te
import sys
import tensorflow as tf


if __name__=='__main__':
    FLAGS=tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("train_sample_num",100000000,"train sample num")   #170000000, 260000000, 58
    tf.app.flags.DEFINE_integer("dev_sample_num",132768,"dev sample num")      
    t_job = te.TensorFlow(entry_point='/media/cfs/zhangjun386/Sugg_rank/ctr_model/train.py',\
                        train_gpu_count=1,
                        hyperparameters={'train_sample_num':FLAGS.train_sample_num, 'dev_sample_num':FLAGS.dev_sample_num}, \
                        )
    t_job.fit(base_job_name='suggest-rank', group='ea-search-qp', cluster='sr-cluster', node_labels=['search-qp-ht','search-qp-a100'], cpu=12, memory=10)
