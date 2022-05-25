import sys
import tensorflow as tf
tf.reset_default_graph()
import numpy as np
from pathlib import Path
from tensorflow.contrib import predictor
from data_util import load_test,ndcg,load_test_ori,load_preprocess

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("test_file","./data/test_top.txt",'path of test file')
tf.app.flags.DEFINE_string("save_file","./data/test_top.predict",'path of test file')
tf.app.flags.DEFINE_string("ckpt_dir","output_03.11/checkpoint","checkpoint location for the model")



def predict():
    f = open(FLAGS.save_file,'w')
    #lines, features  = load_test(FLAGS.test_file, flag=True)
    lines, features  = load_preprocess(FLAGS.test_file)

    print(len(lines))
    subdirs = [x for x in Path(FLAGS.ckpt_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
    model_pb = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(model_pb)
    for i, (line, feature ) in enumerate(zip(lines, features)):
        scores = list()
        result_ori = list()
        for l, fs in zip(line, feature):
            feed_dict = fs
            result = predict_fn(feed_dict)
            score  = result['predictions'][0][0]
            scores.append(score)
            result_ori.append(l)
        # sort
        scores = np.array(scores)
        idxs = np.argsort(-scores)
        for i, idx in enumerate(idxs):
            if i<20:
                f.write(result_ori[idx]+'\t'+str(scores[idx])+'\n')
        f.write('\n')


if __name__=='__main__':
    predict() 
