# -*-coding:utf8 -*-
import sys
import numpy as np


def load():
    f = open('./cid3_name.txt')
    cid3 = list()
    cid3_index = dict()
    for line in f:
        terms = line.strip().split('\t')
        if int(terms[0])-1 not in cid3_index:
            cid3_index[int(terms[0]) -1] = terms[1]+':'+terms[2]
        cid3.append(terms[1]+':'+terms[2])
    return cid3, cid3_index


def load_only():
    f = open('./cid3_name.txt')
    cid3 = list()
    cid3_index = dict()
    for line in f:
        terms = line.strip().split('\t')
        if int(terms[0])-1 not in cid3_index:
            cid3_index[int(terms[0]) -1] = terms[1]
        cid3.append(terms[1])
    return cid3, cid3_index

def similar(txtfile, savefile, top):
    fs = open(savefile,'w')
    cid3, cid3_index = load_only()
    vectors = np.loadtxt(txtfile)
    vec = np.array(vectors)
    similar = np.dot(vec, vec.T)
    diag = np.diag(similar)
    inv_diag = 1 / diag
    # isinf to 0
    inv_diag[np.isinf(inv_diag)] = 0.0
    inv_diag = np.sqrt(inv_diag)
    
    # cosine similar
    cosine_matric = similar * inv_diag
    cosine_matric = cosine_matric.T * inv_diag
    
    for i,x in enumerate(cosine_matric):
        x_sorted = sorted(x, reverse=True)
        index_sorted = np.argsort(-x)
        final = list()
        final_s = list()
        c_s = list()
        for j,(s,idx) in enumerate(zip(x_sorted,index_sorted)):
            if j<int(top):
                final.append(cid3_index[idx]+':'+str(s))
                #final.append(cid3_index[idx])
                #final_s.append(str(s))
                #break 
        #fs.write(cid3[i]+'\t'+','.join(final)+'\t'+','.join(final_s)+'\n')
        fs.write(cid3[i]+'\t'+','.join(final)+'\n')
 


def load_file(filename):
    f = open(filename)
    result = dict()
    for line in f:
        terms = line.strip().split('\t')
        if terms[0] not in result:
            result[terms[0]] = terms[1]
    return result

def load_vec():
    f = open('./cid3.top100.2')
    result = dict()
    for line in f:
        terms = line.strip().split('\t')
        if terms[0] not in result:
            result[terms[0]] = terms[1]+'\t'+terms[2]
    return result


def extract(inf):
    cid, score = [], []
    for x in inf.split(','):
        c,s = x.split(':')
        cid.append(c)
        score.append(float(s))
    return cid, score

def get(c1, c2, similar_matrix):
    data = np.zeros((len(c1),len(c2)))
    for i in range(len(c1)):
        line = similar_matrix[c1[i]]
        cid3, score = line.split('\t')
        cid3_list = cid3.split(',')
        score_list = score.split(',')
        for j in range(len(c2)):
            if c2[j] in cid3_list:
                index = cid3_list.index(c2[j])
                score = score_list[index]
                data[i,j] = float(score)
    return data 
                 
 
def score(query_inf, sku_inf, similar_matrix):
    flag = False
    query_cid, query_score= extract(query_inf) 
    sku_cid, sku_score= extract(sku_inf) 
    # step 1 n*m weight matrix
    query_score = np.array(query_score).reshape(len(query_score),1)
    sku_score = np.array(sku_score).reshape(1, len(sku_score))
    # step 1.2 normalize weight
    if flag:
        sum_query = np.sqrt(np.dot(query_score.T, query_score)) 
        sum_sku = np.sqrt(np.dot(sku_score, sku_score.T))
        query_score = query_score / sum_query
        sku_score = sku_score / sum_sku
    else:
        sum_query = query_score.sum()
        sum_sku = sku_score.sum()
        query_score = query_score / sum_query
        sku_score = sku_score / sum_sku
    # normalize done 
    W = np.dot(query_score, sku_score)
    # step 2 n*m  cid3 and cid3 similar matrix    
    C = get(query_cid, sku_cid, similar_matrix) 
    S = W * C
    # step 3 sum S
    similar = S.sum()
    return similar
 
def QS(queryfile, skufile, savefile):
    query_dict = load_file(queryfile)
    sku_dict = load_file(skufile)
    similar_matrix = load_vec()
    f = open(savefile,'w') 
    for query in query_dict:
        temp = list()
        for sku in sku_dict:
            s = score(query_dict[query], sku_dict[sku], similar_matrix) 
            temp.append(str(s)+'\t'+sku+'\t'+sku_dict[sku])
        f.write(query+'\n'+'\n'.join(temp)+'\n')

 

if __name__=='__main__':
    txtfile, savefile, top= sys.argv[1:]
    similar(txtfile, savefile, top)
    #queryfile, skufile, savefile = sys.argv[1:]
    #QS(queryfile, skufile, savefile) 
