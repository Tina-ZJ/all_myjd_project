import sys
import tensorflow as tf
tf.reset_default_graph()
import numpy as np
from data_util import load_test,create_term
from pathlib import Path
from tensorflow.contrib import predictor

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("term_index","./term_index.txt",'path of term')
tf.app.flags.DEFINE_string("label_index","./cidx_index.txt",'path of cid3')
tf.app.flags.DEFINE_string("test_file","./test.txt",'path of test file')
tf.app.flags.DEFINE_string("ckpt_dir","./checkpoint","checkpoint location for the model")



def predict():
    cid2id, id2cid = create_term(FLAGS.label_index)
    Xinput_ids, Xinput_mask, Xsegment_ids, Xlabels = load_test(FLAGS.test_file)
    print(len(Xinput_ids))
    subdirs = [x for x in Path(FLAGS.ckpt_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
    model_pb = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(model_pb)
    correct = 0.0
    sums = 0.0
    acc = 0.0
    for i, (input_ids, input_mask, segment_ids, labels) in enumerate(zip(Xinput_ids, Xinput_mask, Xsegment_ids, Xlabels)):
        feed_dict = {'input_ids': [input_ids],
                     'input_mask': [input_mask],
                     'segment_ids': [segment_ids]}
        result = predict_fn(feed_dict)
        batch_predictions = result['predictions'] 
        for predictions in batch_predictions:
            cid_set = set()
            predictions_sorted = sorted(predictions, reverse=True)
            index_sorted = np.argsort(-predictions)
            for index, predict in zip(index_sorted, predictions_sorted):
                if len(cid_set) < len(labels):
                    cid = id2cid[index]
                    cid_set.add(int(cid))
            # acc
            same = cid_set & set(labels)
            correct+=len(same)
            sums+=len(cid_set) 
    acc = correct / sums
    print(acc)
    return acc


if __name__=='__main__':
    predict() 
