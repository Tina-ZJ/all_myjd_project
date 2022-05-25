import das.tensorflow.estimator as te
import sys
import tensorflow as tf

if __name__=='__main__':
    FLAGS=tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("train_sample_num",680000000,"train sample num")   #170000000, 260000000, 58
    tf.app.flags.DEFINE_integer("dev_sample_num",1327682,"dev sample num")      
    t_job = te.TensorFlow(entry_point='/media/cfs/zhangjun386/QP_personalization/bert_v2/train_bert.py',\
                        train_gpu_count=2,
                        hyperparameters={'train_sample_num':FLAGS.train_sample_num, 'dev_sample_num':FLAGS.dev_sample_num}, \
                        )
    t_job.fit(base_job_name='qp-personlization-v2', group='ea-search-rank-ht', cluster='sr-cluster', node_labels=['search','mjq','train','search-ht','search-lf', 'expe-cpu-ht','expe-cpu-lf'], cpu=10, memory=20)
