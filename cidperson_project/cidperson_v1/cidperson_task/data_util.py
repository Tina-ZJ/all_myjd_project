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


def chage(content):
    result = content.replace('[','').replace(']','').split(',')
    result = [int(x.replace('"','')) for x in result]
    return result

def load_test(test_file):
    f = open(test_file)
    Xinput_ids, Xinput_mask, Xsegment_ids, Xlabels =[], [], [], []
    for line in f:
        terms = line.strip().split('\t')
        if len(terms)!=6:
            continue
        
        label = chage(terms[4])
        input_ids = chage(terms[1]) 
        input_mask = chage(terms[2])
        segment_ids = chage(terms[3])
        Xinput_ids.append(input_ids) 
        Xinput_mask.append(input_mask) 
        Xsegment_ids.append(segment_ids) 
        Xlabels.append(label)
    return Xinput_ids, Xinput_mask, Xsegment_ids, Xlabels 
             
def create_term(term_file):
    term2idx = {}
    idx2term = {}
    with open(term_file ) as f:
        for line in f:
            line = line.strip('\n')
            if line:
                tokens = line.split('\t')
                if len(tokens) !=2:
                    print (line)
                term2idx[tokens[1]] = int(tokens[0])
                idx2term[int(tokens[0])] = tokens[1]
    return term2idx, idx2term

    
