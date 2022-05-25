import time
import helper
import tensorflow as tf
from tensorflow.contrib.crf.python.ops import crf
import numpy as np
from BILSTM_CRF import BILSTM_CRF
import EntityUtil
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
from functools import wraps


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s : %s seconds" % (function.func_name, str((t1 - t0))))
        return result

    return function_timer

class segClient():
    def __init__(self):
        self.model_path = os.path.join(sys.argv[1],"seg")
        char2id_file = os.path.join(self.model_path, "char2id")
        label2id_file = os.path.join(self.model_path, "label2id")
        self.char2id, self.id2char = helper.loadMap(char2id_file)
        self.label2id, self.id2label = helper.loadMap(label2id_file)
        self.params = helper.loadModelParameters(os.path.join(self.model_path, "param"))
        self.seq_max_len = int(self.params["max_seq_len"])
        name = self.params["name"]
        num_steps = self.seq_max_len
        emb_dim = int(self.params["emb_dim"])
        hidden_dim = int(self.params["hidden_dim"])
        num_layers = int(self.params["num_layers"])
        num_chars = len(self.id2char.keys())
        num_classes = len(self.id2label.keys())

        self.sess = tf.Session()

        initializer = tf.random_uniform_initializer(-0.1,0.1)
        with tf.variable_scope(name,reuse = None, initializer = initializer):
            self.model = BILSTM_CRF(num_chars=num_chars,num_classes=num_classes,num_steps=num_steps,
                                        hidden_dim=hidden_dim,num_layers=num_layers,emb_dim=emb_dim,
                                       batch_size=1,is_training=False)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.model_path, "model"))

    def predict(self, x):
        results = []
        logits, transition_params, sequence_lengths = \
            self.sess.run([
                self.model.logits, self.model.transition_params, self.model.length],
                feed_dict={
                    self.model.inputs: x,
                })
        for logit, _seq_len in zip(logits, sequence_lengths):
            if _seq_len > 0:
                score = logit[:_seq_len]
                viterbi_sequence, _ = crf.viterbi_decode(score, transition_params)
                results.append(viterbi_sequence)
            else:
                results.append([])
        return results


    def test(self,line):
        new_char = self.char2id["<NEW>"]
        line_ids = [self.char2id[ch] if ch in self.char2id else new_char for ch in line]
        if len(line_ids) <= self.seq_max_len:
            line_ids.extend([0] * (self.seq_max_len - len(line_ids)))
        line_ids = np.array([line_ids])
        label_ids = self.predict(line_ids)

        tokens = EntityUtil.parseTag(label_ids[0],self.id2label, line)
        return tokens

if __name__ == '__main__':
    if len(sys.argv) !=2:
        print "input %s model" % (sys.argv[0])
        sys.exit()
    client = segClient()
    line = "oppo"
    seg_result = client.test(line)
    line = raw_input('please input a query:')
    while line != 'q':
        if line == '':
            line = raw_input('please input the keyword')
            continue
        print("query:" + line)
        seg_result = client.test(line.decode('utf8'))
        print("the seg result:" + ' '.join(seg_result))
        line = raw_input('please input the query:')
    print("program exit")
