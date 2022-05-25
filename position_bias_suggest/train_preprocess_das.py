import das.tensorflow.estimator as te
import sys
import tensorflow as tf

group_1='ea-search-rank-online'
node_labels_1=['search-prod-spu-cpu-lf','search-online-zyx']


group_2 ='ea-search-rank-jdc'
node_labels_2=['search-lf','search','mjq','lf','ht','search-ht']

group_3='ea-search-rank-ht'
node_labels_3=['search-lf','search','mjq','train','search-ht']

if __name__=='__main__':
    FLAGS=tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("train_sample_num",600000000,"train sample num")   #170000000, 260000000, 58
    tf.app.flags.DEFINE_integer("dev_sample_num",132768,"dev sample num")      
    t_job = te.TensorFlow(entry_point='/media/cfs/zhangjun386/Sugg_rank/train_preprocess.py',\
                        train_gpu_count=1,
                        hyperparameters={'train_sample_num':FLAGS.train_sample_num, 'dev_sample_num':FLAGS.dev_sample_num}, \
                        )
    t_job.fit(base_job_name='suggest-rank', group=group_1, cluster='sr-cluster', node_labels=node_labels_1, cpu=12, memory=10)
