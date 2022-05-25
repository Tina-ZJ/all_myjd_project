# -*- coding: utf-8 -*-
import codecs
import numpy as np
import os
import pickle
import codecs

def get_embs(embfile,word2index_pre, index2word, vocab_size, embed_size=128):
    data = np.loadtxt(embfile,dtype=np.float32)
    pre_embs = [[]] * vocab_size
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    for i in range(vocab_size):
        if i in index2word:
            word = index2word[i]
            if word in word2index_pre:
                idx = word2index_pre[word]
                pre_embs[i] = data[idx,:]
            else:
                pre_embs[i] = np.random.uniform(-bound, bound, embed_size) 
        else: 
            pre_embs[i] = np.random.uniform(-bound, bound, embed_size) 
    word_embs = np.array(pre_embs)
    print(word_embs.shape)
    return word_embs 

            
def create_term(term_file):
    term2idx = {}
    idx2term = {}
    with open(term_file ) as f:
        for line in f:
            line = line.replace('\r\n','\n')
            line = line.strip('\n')
            if line:
                tokens = line.split('\t')
                if len(tokens) !=2:
                    print (line)
                term2idx[tokens[1]] = int(tokens[0])
                idx2term[int(tokens[0])] = tokens[1]
    return term2idx, idx2term

        
def create_label(label_file):
    label2idx = {}
    idx2label = {}
    with open(label_file) as f:
        for i, line in enumerate(f):
            line = line.replace('\r\n','\n')
            line = line.strip()
            if line:
                tokens = line.split('\t')
                label2idx[tokens[0]] = int(tokens[1])
                idx2label[int(tokens[1])] = tokens[0]
    return label2idx, idx2label


def load_test(testfile, word2id):
    f = open(testfile)
    data = list()
    query_ids_list = list()
    sku_ids_list = list()
    query_segs_list = list()
    sku_segs_list = list()
    for line in f:
        terms = line.strip().split('\t')
        query_segs = terms[1].split(',')
        sku_segs = terms[-1].split(',')
        query_segs = [x for x in query_segs if x.strip()!='']
        sku_segs = [x for x in sku_segs if x.strip()!='']
        query_ids = [word2id.get(x, 1) for x in query_segs]
        sku_ids = [word2id.get(x, 1) for x in sku_segs]
        query_ids_list.append(query_ids)
        sku_ids_list.append(sku_ids)
        data.append(terms[0]+'\t'+terms[2])
        query_segs_list.append(query_segs)
        sku_segs_list.append(sku_segs)
    return data, query_ids_list, sku_ids_list, query_segs_list, sku_segs_list 
