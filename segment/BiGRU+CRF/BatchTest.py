import time
import math
import helper
import argparse
import tensorflow as tf
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


def parser_paramter():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="the path of the model file")
    parser.add_argument("test_path", help="the path of the test file")
    parser.add_argument("output_path", help="the path of the output")
    parser.add_argument("-bs", "--batch_size", help="the batch size, the default is 200", default=200, type=int)
    parser.add_argument("-xpu", "--xpu", help="the configure of gpu or cpu, the default is /gpu:0", default="/gpu:0",
                        type=str)

    args = parser.parse_args()
    model_path = args.model_path
    test_path = args.test_path
    output_path = args.output_path
    xpu_config = args.xpu
    batch_size = args.batch_size
    return (model_path, test_path, output_path, xpu_config, batch_size)


@fn_timer
def Batch_Test():
    (model_path, test_path, output_path, xpu_config, batch_size) = parser_paramter()
    char2id_file = os.path.join(model_path, "char2id")
    label2id_file = os.path.join(model_path, "label2id")

    print "preparing test data"
    params = helper.loadModelParameters(os.path.join(model_path, "param"))
    name = params["name"]
    num_steps = int(params["max_seq_len"])
    emb_dim = int(params["emb_dim"])
    hidden_dim = int(params["hidden_dim"])
    num_layers = int(params["num_layers"])

    char2id, id2char = helper.loadMap(char2id_file)
    label2id, id2label = helper.loadMap(label2id_file)
    num_chars = len(id2char.keys())
    num_classes = len(id2label.keys())
    x_test, x_test_txt, splits_per_line = helper.getTest(test_path, char2id, seq_max_len=num_steps)
    print "test size: %d." % len(x_test)
    print "building model"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3) 
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope(name, reuse=None, initializer=initializer):
            model = BILSTM_CRF(num_chars=num_chars, num_classes=num_classes, num_steps=num_steps,
                               hidden_dim=hidden_dim, num_layers=num_layers, emb_dim=emb_dim,
                               batch_size=batch_size, is_training=False)

        print "loading model parameter"
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_path, "model"))

        print "testing"
        start_time = time.time()
        with tf.device(xpu_config):
            test(model, sess, x_test, x_test_txt, splits_per_line, output_path, id2label, name)

        end_time = time.time()
        print "test time used %f(seconds)" % (end_time - start_time)


def test(model, sess, x_test, x_test_txt, splits_per_line, output_path, id2label, test_type):
    batch_size = model.batch_size
    num_iterations = int(math.ceil(1.0 * len(x_test) / batch_size))
    print "number of iteration: " + str(num_iterations)
    txt_index = 0
    prev_buf = []
    with open(output_path, "wb") as outfile:
        for i in range(num_iterations):
            print "iteration: " + str(i + 1)
            x_test_batch = x_test[i * batch_size: i * batch_size + batch_size]
            cur_size = batch_size
            if i == num_iterations - 1 and len(x_test_batch) < batch_size:
                cur_size = len(x_test_batch)
                zero_vec = [0] * model.num_steps
                x_test_batch.extend([zero_vec] * (batch_size - cur_size))

            x_test_batch = np.array(x_test_batch)
            label_ids = model.predictBatch(sess, x_test_batch)
            if cur_size < batch_size:
                label_ids = label_ids[:cur_size]

            txt_index, prev_buf = decode(prev_buf, label_ids, txt_index,
                                         x_test_txt, splits_per_line, outfile, id2label)
        outfile.flush()
        outfile.close()


def decode(prev_buf, label_ids, txt_index, x_test_txt, splits_per_line, outfile, id2label):
    # for ids in label_ids:
    #     t = " ".join([id2label[id] for id in ids])
    #     outfile.write(t+"\n")
    if len(prev_buf) + len(label_ids) < splits_per_line[txt_index]:
        prev_buf.extend(label_ids)
        return txt_index, prev_buf
    ids = []
    for i in range(len(prev_buf)):
        ids.extend(prev_buf[i])
    for i in range(0, splits_per_line[txt_index] - len(prev_buf)):
        ids.extend(label_ids[i])
    decodeOneLine(ids, id2label, x_test_txt[txt_index], outfile)

    id_index = splits_per_line[txt_index] - len(prev_buf)
    txt_index += 1
    prev_buf = []
    while txt_index < len(splits_per_line) and len(label_ids) - id_index >= splits_per_line[txt_index]:
        ids = []
        for i in range(id_index, id_index + splits_per_line[txt_index]):
            ids.extend(label_ids[i])
        decodeOneLine(ids, id2label, x_test_txt[txt_index], outfile)
        id_index += splits_per_line[txt_index]
        txt_index += 1
    for i in range(id_index, len(label_ids)):
        prev_buf.append(label_ids[i])
    return txt_index, prev_buf


def decodeOneLine(ids, id2label, text, outfile):
    #tokens = [id2label[i] for i in ids]
    #outfile.write(" ".join(tokens) + "\n")
    tokens = EntityUtil.parseTag(ids[:len(text)], id2label, text)
    outfile.write(" ".join(tokens) + "\n")


if __name__ == '__main__':
    Batch_Test()
