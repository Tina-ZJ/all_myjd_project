# -*-coding:utf8 -*-
import sys
import numpy as np
from sklearn.neighbors import KDTree

def load_query():
    f = open('./Query/query.txt')
    querys = list()
    for query in f:
        querys.append(query.strip())

    return querys


def top(savefile, topN):
    f = open(savefile, 'w')
    querys = load_query()
    cid3, cid3_index = load() 
    query_emb = np.loadtxt('./query.emb')
    query_emb = np.array(query_emb) 
    cid_emb = np.loadtxt('./cid3.emb')
    cid_emb = np.array(cid_emb)
    tree = KDTree(cid_emb, leaf_size=20)
    #print(cid_emb.shape) 
    #print(query_emb.shape) 
    for i,x in enumerate(query_emb):
        #print(x.shape)
        x = x.reshape(1,-1)
        #print(x.shape)
        dist, ind = tree.query(x, k=int(topN))
        result = list()
        for y,d in zip(ind[0], dist[0]):
            #print(y)
            cos_similar = (2-d*d)/2
            result.append(cid3_index[y]+':'+str(cos_similar)) 
        f.write(querys[i]+'\t'+','.join(result)+'\n') 

def load():
    f = open('./Cid3/cid3_feature.txt')
    cid3 = list()
    cid3_index = dict()
    for i,line in enumerate(f):
        terms = line.strip().split('\t')
        cid3_index[i] = terms[0]+':'+terms[1]
        cid3.append(terms[0]+':'+terms[1])
    return cid3, cid3_index

if __name__=='__main__':
    savefile, topN= sys.argv[1:]
    top(savefile, topN)
