# coding:utf-8

import os
import csv
import re
import numpy as np
import pandas as pd
import fileinput
import random
import sys

reload(sys)
sys.setdefaultencoding('utf8')


def loadMap(token2id_filepath):
    token2id = {}
    id2token = {}
    with open(token2id_filepath) as infile:
        for row in infile:
            row = row.rstrip().decode("utf-8")
            token = row.split(' ')[0]
            token_id = int(row.split(' ')[1])
            token2id[token] = token_id
            id2token[token_id] = token
    return token2id, id2token


def saveMap(id2Token, file):
    with open(file, "wb") as outfile:
        for idx in id2Token.keys():
            outfile.write(id2Token[idx] + " " + str(idx) + "\n")


def buildMap(train_path, char2id_file='char2id', label2id_file='label2id'):
    chars = set()
    labels = set()
    for line in fileinput.input(train_path):
        line = line.strip().decode("utf-8")
        for token in line.split("\t"):
            idx = token.rfind("/")
	    if idx >=0 :
	        char = token[:idx]
		label = token[idx + 1:]
                chars.add(char)
                labels.add(label)
    chars = list(chars)
    labels = list(labels)
    char2id = dict(zip(chars, range(1, len(chars) + 1)))
    label2id = dict(zip(labels, range(1, len(labels) + 1)))
    id2char = dict(zip(range(1, len(chars) + 1), chars))
    id2label = dict(zip(range(1, len(labels) + 1), labels))
    id2char[0] = "<PAD>"
    id2label[0] = "<PAD>"
    char2id["<PAD>"] = 0
    label2id["<PAD>"] = 0
    id2char[len(chars) + 1] = "<NEW>"
    char2id["<NEW>"] = len(chars) + 1

    saveMap(id2char, char2id_file)
    saveMap(id2label, label2id_file)

    return char2id, id2char, label2id, id2label


def saveModelParameters(save_path, params):
    nw = open(os.path.join(save_path, "param"), 'w')
    for key, value in params:
        nw.write(key + "=" + str(value) + "\n")
    nw.flush()
    nw.close()


def loadModelParameters(save_path):
    params = {}
    for line in fileinput.input(save_path):
        k, v = _parseProperty(line)
        if len(k) > 0 and len(v) > 0:
            params[k] = v
    return params


def _parseProperty(line):
    if line.find("#") >= 0:
        line = line[0:line.find('#')]
    if line.find('=') > 0:
        strs = line.split('=')
        return strs[0].strip(), strs[1].strip()
    if line.find(':') > 0:
        strs = line.split(':')
        return strs[0].strip(), strs[1].strip()
    return line.strip(), ""


def getTrain(train_path, char2id, label2id, seq_max_len, train_val_ratio=0.95):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    for line in fileinput.input(train_path, bufsize=10240):
        line = line.strip().decode("utf-8")
        if random.random() < train_val_ratio:
            _parseTrainLine(line, x_train, y_train, char2id, label2id, seq_max_len)
        else:
            _parseTrainLine(line, x_val, y_val, char2id, label2id, seq_max_len)

    print "train size: %d, validation size: %d" % (len(x_train), len(y_val))

    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)


def _parseTrainLine(line, x, y, char2id, label2id, max_seq_len):
    newchar = char2id["<NEW>"]
    padlabel = label2id["<PAD>"]
    punc1 = set([u'。', u'；', u'！', u'!', u';', u'？', u'?'])
    punc2 = set([u'，', u','])
    charids = []
    labels = []
    ind1 = -1
    ind2 = -1
    for ind, token in enumerate(line.split("\t")):
        idx = token.rfind('/')
        if idx >= 0:
            t = token[:idx]
            l = token[idx + 1:]
            charids.append(char2id[t] if t in char2id else newchar)
            labels.append(label2id[l] if l in label2id else padlabel)
            if l not in label2id:
                print token
                print line
            if t in punc1:
                ind1 = len(charids)
            if t in punc2:
                ind2 = len(labels)
            if len(charids) >= max_seq_len:
                if ind1 >= 0:
                    _appendTrainLine(x, y, charids[:ind1], labels[:ind1], max_seq_len)
                    charids = charids[ind1:]
                    labels = labels[ind1:]
                elif ind2 >= 0:
                    _appendTrainLine(x, y, charids[:ind2], labels[:ind2], max_seq_len)
                    charids = charids[ind2:]
                    labels = labels[ind2:]
                else:
                    _appendTrainLine(x, y, charids, labels, max_seq_len)
                    charids = []
                    labels = []
                ind1 = -1
                ind2 = -1
    if len(charids) > 0:
        _appendTrainLine(x, y, charids, labels, max_seq_len)


def _appendTrainLine(x, y, charids, labels, max_seq_len):
    length = len(charids)
    charids.extend([0] * (max_seq_len - length))
    labels.extend([0] * (max_seq_len - length))
    x.append(charids)
    y.append(labels)


def getEmbedding(infile_path, char2id_file):
    char2id, id2char = loadMap(char2id_file)
    firstLine = True
    emb_ext = []
    inc_ext = len(char2id.keys())
    assert (inc_ext not in id2char and inc_ext - 1 in id2char)
    for row in fileinput.input(infile_path, bufsize=10240):
        row = row.strip().decode("utf-8")
        if firstLine:
            firstLine = False
            num_chars = int(row.split()[0])
            emb_dim = int(row.split()[1])
            emb_matrix = np.zeros((len(char2id.keys()), emb_dim))
            continue
        items = row.split()
        char = items[0]
        emb_vec = [float(val) for val in items[1:]]
        if char in char2id:
            emb_matrix[char2id[char]] = emb_vec
        else:
            id2char[inc_ext] = char
            emb_ext.append(emb_vec)
            inc_ext += 1
    # emb_matrix
    if inc_ext > len(char2id.keys()):
        emb_matrix = np.vstack((emb_matrix, np.array(emb_ext)))
    saveMap(id2char, char2id_file)
    return emb_matrix


def nextBatch(X, y, start_index, batch_size=128):
    last_index = start_index + batch_size
    X_batch = list(X[start_index:min(last_index, len(X))])
    y_batch = list(y[start_index:min(last_index, len(y))])
    if last_index > len(X):
        left_size = last_index - (len(X))
        for i in range(left_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch


def nextRandomBatch(X, y, batch_size=128):
    X_batch = []
    y_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(X))
        X_batch.append(X[index])
        y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch


def getTest(test_path, char2id, seq_max_len=200):
    x_test = []
    x_test_txt = []
    splits_per_line = []
    new_char = char2id["<NEW>"]
    punc1 = set([u'。', u'；', u'！', u'!', u';', u'？', u'?'])
    punc2 = set([u'，', u','])

    for line in fileinput.input(test_path):
        line = line.strip().decode("utf-8")
        if len(line) == 0:
            continue
        line_ids = [char2id[ch] if ch in char2id else new_char for ch in line]
        if len(line_ids) <= seq_max_len:
            line_ids.extend([0] * (seq_max_len - len(line_ids)))
            x_test.append(line_ids)
            x_test_txt.append(line)
            splits_per_line.append(1)
        else:
            splits = splitOneLine(line_ids, line, punc1, punc2, seq_max_len)
            x_test.extend(splits)
            x_test_txt.append(line)
            splits_per_line.append(len(splits))
    return x_test, x_test_txt, splits_per_line


def splitOneLine(ids, str, punc1, punc2, max_seq_len):
    l = len(ids)
    results = []
    line = []
    idx1 = -1
    idx2 = -1
    for i in range(l):
        line.append(ids[i])
        c = str[i]
        if c in punc1:
            idx1 = len(line)
        if c in punc2:
            idx2 = len(line)
        if len(line) >= max_seq_len:
            if idx1 >= 0:
                appendTestLine(results, line[:idx1], max_seq_len)
                line = line[idx1:]
            elif idx2 >= 0:
                appendTestLine(results, line[:idx2], max_seq_len)
                line = line[idx2:]
            else:
                appendTestLine(results, line, max_seq_len)
                line = []
            idx1 = -1
            idx2 = -1
    if len(line) > 0:
        appendTestLine(results, line, max_seq_len)
    return results


def appendTestLine(results, charids, max_seq_len):
    charids.extend([0] * (max_seq_len - len(charids)))
    results.append(charids)
