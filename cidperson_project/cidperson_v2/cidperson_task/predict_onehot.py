import sys
import tensorflow as tf
tf.reset_default_graph()
import numpy as np
from data_util import load_test,create_term,load_test_v2, load_test_v3
from pathlib import Path
from tensorflow.contrib import predictor

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("label_name","data/valid_cids.name",'path of cid3')
tf.app.flags.DEFINE_string("term_index","data/term_index_10_07.txt",'path of term')  #term_index_6_12_v6.txt
tf.app.flags.DEFINE_string("valid_cids","data/valid_cids.name",'path of term')
tf.app.flags.DEFINE_string("label_index","data/cidx_index_v1.txt",'path of cid3')
tf.app.flags.DEFINE_string("test_file","./test.txt",'path of test file')
tf.app.flags.DEFINE_string("save_file","./test.dnn",'path of save file')
tf.app.flags.DEFINE_string("ckpt_dir","output_4L_gelu/checkpoint","checkpoint location for the model")



def predict():
    f = open(FLAGS.save_file,'w')
    word2id, id2word = create_term(FLAGS.term_index)
    cid2id, id2cid = create_term(FLAGS.label_index)
    name2cid, cid2name = create_term(FLAGS.valid_cids)
    Xinput_ids, Xinput_mask, Xsegment_ids, lines = load_test_v3(FLAGS.test_file, word2id, id2word)
    print(len(Xinput_ids))
    subdirs = [x for x in Path(FLAGS.ckpt_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
    model_pb = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(model_pb)
    for i, (input_ids, input_mask, segment_ids) in enumerate(zip(Xinput_ids, Xinput_mask, Xsegment_ids)):
        feed_dict = {'input_ids': [input_ids],
                     'input_mask': [input_mask]}
        result = predict_fn(feed_dict)
        batch_predictions = result['predictions'] 
        for predictions in batch_predictions:
            cid_list = []
            predictions_sorted = sorted(predictions, reverse=True)
            index_sorted = np.argsort(-predictions)
            for index, predict in zip(index_sorted, predictions_sorted):
                if predict >0.01 or len(cid_list)<3:
                    cid = id2cid[index]
                    name = cid2name.get(int(cid), '未知')
                    cid_list.append(cid+':'+name+':'+str(predict))
                 
            f.write(lines[i]+'\t'+','.join(cid_list)+'\n')




if __name__=='__main__':
    predict() 
