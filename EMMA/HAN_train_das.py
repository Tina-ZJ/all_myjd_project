import das.tensorflow.estimator as te
import sys
import tensorflow as tf

if __name__=='__main__':
    FLAGS=tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("num_classes_first",50,"number of label")
    tf.app.flags.DEFINE_integer("num_classes_second",500,"number of label")
    tf.app.flags.DEFINE_integer("num_classes_product",500,"number of label")
    tf.app.flags.DEFINE_integer("num_classes_brand",500,"number of label")
    tf.app.flags.DEFINE_integer("train_sample_num",100000000,"train sample num")
    tf.app.flags.DEFINE_integer("dev_sample_num",1327682,"dev sample num")      
    t_job = te.TensorFlow(entry_point='/media/cfs/zhangjun386/Multi-Task-pipline_new3_seg3/HAN_hierarchical_multi_task_intention_new/HAN_train_v4.py',\
                        train_gpu_count=1,
                        hyperparameters={'num_classes_first':FLAGS.num_classes_first, 'num_classes_second':FLAGS.num_classes_second, 'num_classes_product':FLAGS.num_classes_product, 'num_classes_brand':FLAGS.num_classes_brand, 'train_sample_num':FLAGS.train_sample_num, 'dev_sample_num':FLAGS.dev_sample_num}, \
                        )
    t_job.fit(base_job_name='hirechal-han-task-v4', cluster='ht1', cpu=20, memory=40)
