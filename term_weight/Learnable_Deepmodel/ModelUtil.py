import tensorflow as tf
import numpy as np
import fileinput
import os
import pickle as pick
import sys

def extract_parameter(modelPath):
    nw = open(os.path.join(modelPath, "weight.names"), 'w')
    new_checkpoint = tf.train.latest_checkpoint(modelPath)
    out_local_w = open(os.path.join(modelPath, "local.w"), "w")
    out_local_b = open(os.path.join(modelPath, "local.b"), "w")
    out_global_w = open(os.path.join(modelPath, "global.w"), "w")
    out_global_b = open(os.path.join(modelPath, "global.b"), "w")
    saver = tf.train.import_meta_graph(new_checkpoint+'.meta')
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device("/cpu:0"):
            saver.restore(sess, new_checkpoint)
            for var in tf.trainable_variables():
                data = sess.run(var)
                name = str(var.name).split("/")[-1]
                nw.write(name+" " + str(data.shape) + "\n")
                if 'local_w' in name:
                    np.savetxt(out_local_w, data, fmt='%s', newline='\n')
                if 'local_b' in name:
                    #data = data.T
                    np.savetxt(out_local_b, data, fmt='%s', newline='\n')
                if 'global_w' in name:
                    np.savetxt(out_global_w, data, fmt='%s', newline='\n')
                if 'global_b' in name:
                    np.savetxt(out_global_b, data, fmt='%s', newline='\n')
    #nw.flush()
    #nw.close()


def save_pkl(modelPath):
    parames = dict()
    sum = 0
    saver = tf.train.import_meta_graph(os.path.join(modelPath, "model_ckpt-18.meta"))
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        saver.restore(sess, os.path.join(modelPath, "model"))
        for var in tf.trainable_variables():
            sum += 1
            data = sess.run(var)
            name = var.name
            if sum == 1 or sum == len(tf.trainable_variables()) or sum == len(
                    tf.trainable_variables()) - 1 or sum == len(tf.trainable_variables()) - 2:
                continue
            parames[name] = data
        f = file(os.path.join(modelPath, "params.pkl"), 'wb')
        pick.dump(parames, f, True)
        f.close()


def read_pkl(modelPath):
    f2 = file(os.path.join(modelPath, "params.pkl"), 'rb')
    result = pick.load(f2)
    f2.close()
    return result

if __name__ == "__main__":
    path = sys.argv[1]
    extract_parameter(path)
