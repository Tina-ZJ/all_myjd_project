import das.tensorflow.estimator as te
import sys
import tensorflow as tf

if __name__=='__main__':
    #FLAGS=tf.app.flags.FLAGS
    #tf.app.flags.DEFINE_integer("dev_sample_num",1327682,"dev sample num")      
    t_job = te.TensorFlow(entry_point='/media/cfs/zhangjun386/alBert/albert_multi_task/run_sequencelabeling.py',\
                        train_gpu_count=1,\
                        )
    t_job.fit(base_job_name='das-bert-tag-multi-task')
