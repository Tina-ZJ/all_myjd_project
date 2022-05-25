import das.tensorflow.estimator as te
import sys
import tensorflow as tf

if __name__=='__main__':
    erp = sys.argv[1]
    train_path = sys.argv[2]
    FLAGS=tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("train_sample_num",600000000,"train sample num")
    t_job = te.TensorFlow(entry_point='/media/cfs/%s/QueryRootData/src/qp_apps/applications/cidperson_task/train_bert.py' % (erp), \
                        train_gpu_count=1,
                        hyperparameters={'train_sample_num':FLAGS.train_sample_num, 'train_path':train_path}, \
                        )
    t_job.fit(base_job_name='qp-personlization', group='ea-search-rank-ht', cluster='sr-cluster', node_labels=['search','mjq','train','search-ht','search-lf'], cpu=8, memory=20)
