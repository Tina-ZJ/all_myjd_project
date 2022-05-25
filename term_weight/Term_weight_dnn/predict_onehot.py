import sys
import tensorflow as tf
tf.reset_default_graph()
import numpy as np
from data_util import load_test,create_term
from pathlib import Path
from tensorflow.contrib import predictor

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("term_index","data/term_index.txt",'path of term')  #
tf.app.flags.DEFINE_string("test_file","./test.txt",'path of test file')
tf.app.flags.DEFINE_string("save_file","./test.predict",'path of test file')
tf.app.flags.DEFINE_string("ckpt_dir","output/checkpoint","checkpoint location for the model")


def combine(terms, weights):
    result = list()
    result_dict = dict()
    result_2 = list()
    for t, w in zip(terms, weights):
        result.append(t+':'+str(w))
        result_dict[t] = w
    # top5
    weight_sort = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (t,w) in enumerate(weight_sort):
        if i<=5:
            result_2.append(t+':'+str(w)) 
        
    return result, result_2 

def predict():
    f = open(FLAGS.save_file,'w')
    word2id, id2word = create_term(FLAGS.term_index)
    lines, query_ids_list, sku_ids_list, query_segs_list, sku_segs_list = load_test(FLAGS.test_file, word2id)
    print(len(query_ids_list))
    subdirs = [x for x in Path(FLAGS.ckpt_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
    model_pb = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(model_pb)
    for i, (query_ids, sku_ids, query_segs, sku_segs) in enumerate(zip(query_ids_list, sku_ids_list, query_segs_list, sku_segs_list)):
        feed_dict = {'input_query': [query_ids],
                     'pos_item': [sku_ids]}
        result = predict_fn(feed_dict)
        query_weight  = result['query_weight']
        item_weight = result['pos_weight']
        score = result['pos_similar']
        result, result_top = combine(query_segs, query_weight[0])
        result2, result_top2 = combine(sku_segs, item_weight[0])
        f.write(lines[i]+'\n'+','.join(result)+'\t'+','.join(result_top)+'\n'+','.join(result2)+'\t'+','.join(result_top2)+'\n'+str(score[0])+'\n\n')




if __name__=='__main__':
    predict() 
