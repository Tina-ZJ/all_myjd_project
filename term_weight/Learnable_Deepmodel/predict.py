import sys
import tensorflow as tf
tf.reset_default_graph()
import numpy as np
from pathlib import Path
from tensorflow.contrib import predictor
from data_util import load_test,ndcg,load_test_ori

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("test_file","./test.txt",'path of test file')
tf.app.flags.DEFINE_string("save_file","./test.predict",'path of test file')
tf.app.flags.DEFINE_string("ckpt_dir","output/checkpoint","checkpoint location for the model")



def predict():
    f = open(FLAGS.save_file,'w')
    #lines, features, idxs, words = load_test(FLAGS.test_file, flag=True)
    lines, features, idxs, words = load_test_ori(FLAGS.test_file, flag=True)

    print(len(lines))
    subdirs = [x for x in Path(FLAGS.ckpt_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
    model_pb = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(model_pb)
    all_ndcg = list()
    for i, (line, feature, idx, word) in enumerate(zip(lines, features, idxs, words)):
        feed_dict = feature
        result = predict_fn(feed_dict)
        score  = result['predictions']

        # get ndcg
        #pred_idx = np.argsort(score[0][:len(idx)])[::-1]
        pred_idx = np.argsort(score[0])[::-1]
        #pred_idx_= np.array(idx)[pred_idx]
        #ndcg_val = ndcg(pred_idx_)
        
        # get topk terms
        topk_words = list()
        if len(score[0])!=40:
            print (i)
        
        for j in pred_idx[:5]:
            print(j)
            print(len(word))
            topk_words.append(word[j]+':'+str(score[0][j]))
             
        #all_ndcg.append(ndcg_val)
        score = [y+':'+str(x) for x,y in zip(score[0], word)]
        #f.write(line+'\t'+','.join(topk_words)+'\t'+','.join(score)+'\t'+str(ndcg_val)+'\n')
        f.write(line+'\t'+','.join(topk_words)+'\t'+','.join(score)+'\n')
    #print(np.nanmean(all_ndcg))


if __name__=='__main__':
    predict() 
