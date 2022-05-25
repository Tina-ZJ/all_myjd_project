# -*- coding: utf-8 -*-
import codecs
import numpy as np
import os
import pickle
import codecs


def transfor_idx(data_str, float_flag=False):
    data_ids = data_str.replace('[','').replace(']','').split(',')
    if float_flag:
        data_ids = [float(x) for x in data_ids]
    else: 
        data_ids = [int(x) for x in data_ids]
    return data_ids

def pad(data, padding=8, value=0.0):
    data_ids = data[:padding]
    for i in range(0, padding - len(data_ids)):
        data_ids.append(value)
    return data_ids

 
def freq(terms):
    count = dict()
    for t in terms:
        if t not in count:
            count.setdefault(t,0)
        count[t]+=1
    term_freq = [count[t] for t in terms]
    term_freq = pad(term_freq, padding=40, value=0)
    return term_freq 

def load_test_ori(testfile, flag=True):
    f = open(testfile)
    term2idx = load_term('./data/term_index.txt')
    cidterm2f = load_f('./data/term_features.txt')
    lines = list()
    features = list()
    words = list()
    for i, line in enumerate(f):
        terms = line.strip().split('\t')
        word = terms[-1].split(',')
        words.append(word)
        lines.append(terms[-2])
        word = [term2idx.get(x, 1) for x in word]
        input_ids = pad(word, padding=40, value=0)
        input_mask = [1 if x>0 else 0 for x in input_ids]
        segment_ids = freq(word)
        if i==5:
            print(word)
            print(input_ids)
            print(input_mask)
            print(segment_ids)
        if flag: 
            f = {
                'input_ids': [input_ids],
                'input_mask': [input_mask],
                'segment_ids': [segment_ids]
                }
        else:
            input_features = []
            cid = terms[0]
            termf = cidterm2f.get(cid,None)
            if termf==None:
                input_features = [0.0]*len(input_ids)*6
            else:
                for x in word:
                    xf = termf.get(x,[0.0]*6)
                    input_features+=xf
                # pad
                input_features = pad(input_features, padding=240, value=-1.0)
 
            f = {
                 'input_features': [input_features],
                 'input_ids': [input_ids],
                 'input_mask': [input_mask],
                 'segment_ids': [segment_ids]
                }
        features.append(f)

    return lines, features, [1]*len(lines), words

 
def load_term(filename):
    f = open(filename)
    data = dict()
    for line in f:
        terms = line.strip('\n').split('\t')
        data.setdefault(terms[1], int(terms[0]))
    print(len(data))
    return data


def load_f(filename):
    f = open(filename)
    data = dict()
    for line in f:
        terms = line.strip('\n').split('\t')
        features = [float(x) for x in terms[3:]]
        if terms[0] not in data:
            data[terms[0]] = dict()
        data[terms[0]].setdefault(terms[2],features)
    return data

   
def load_test(testfile, flag=False):
    f = open(testfile)
    lines = list()
    features = list()
    group_id = dict()
    features_tmp = list()
    lines_tmp = list()
    for i, line in enumerate(f):
        terms = line.strip().split('\t')
        group = terms[0]
        if group not in group_id:
            if i!=0:
                features.append(features_tmp)
                lines.append(lines_tmp)
            lines_tmp = list()
            features_tmp = list()
            group_id[group]=0
            
        title = '\t'.join(terms[1:3])
        input_ids = transfor_idx(terms[5])    
        input_mask = transfor_idx(terms[6])    
        segment_ids = transfor_idx(terms[7])    
        # dict
        f = {
            'input_ids': [input_ids],
            'input_mask': [input_mask],
            'segment_ids': [segment_ids]
            }
        features_tmp.append(f) 
        lines_tmp.append(title)
    return lines, features



def dcg(scores):
    return np.sum([(np.power(2, scores[i]) - 1) / np.log2(i + 2) for i in range(len(scores))])


def idcg(scores):
    scores = [score for score in sorted(scores)[::-1]]
    return dcg(scores)



def ndcg(predict):
    dcg_val = dcg(predict)
    idcg_val =idcg(predict)
    ndcg_val = (dcg_val / idcg_val)
    return ndcg_val


    
    
