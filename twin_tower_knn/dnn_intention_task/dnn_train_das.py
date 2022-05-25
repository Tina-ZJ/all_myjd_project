import das.tensorflow.estimator as te
import sys
import tensorflow as tf

if __name__=='__main__':
    FLAGS=tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("num_classes",10,"number of label")
    tf.app.flags.DEFINE_integer("train_sample_num",273237386,"train sample num")
    t_job = te.TensorFlow(entry_point='/media/cfs/zhangjun386/KNN/DNN_intention_task/dnn_train.py',\
                        train_gpu_count=1,
                        hyperparameters={'num_classes':FLAGS.num_classes, 'train_sample_num':FLAGS.train_sample_num}, \
                        )
    t_job.fit(base_job_name='das-dnn-intention-task')
