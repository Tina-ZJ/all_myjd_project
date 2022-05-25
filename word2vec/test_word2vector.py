# -*-coding: utf8 -*-
import sys
import gensim
import logging
import os.path
import multiprocessing
import json

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def train():
    # check and process input arguments
    if len(sys.argv) < 3:
        print((globals()['__doc__'] % locals()))
        sys.exit(1)
    inp, outp1, outp_vec = sys.argv[1:4]
    model = Word2Vec(LineSentence(inp), size=300, window=3, min_count=1, workers=multiprocessing.cpu_count())
    # save
    model.save(outp1)
    model.wv.save_word2vec_format(outp_vec,binary=False)

def test(model_path):
    model = Word2Vec.load(model_path)
    # test most similar words
    #sim = model.wv.most_similar(u'女人')
    #for s in sim :
        #print(s[0],s[1])
    result = model.most_similar(u'手机卡')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'公主床')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'运动鞋')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'充电器')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'花花公子')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'手机套')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'苹果')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'真皮')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'黑色')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'平底锅')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'拖鞋')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'老板椅')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    result = model.most_similar(u'欧莱雅')
    print(json.dumps(result, encoding='utf-8', ensure_ascii=False))
    # test does not match
    result = model.doesnt_match(u"对讲机 电台 耳机".split())
    print(result)
    result = model.similarity(u'对讲机', u"耳机")
    print(result)
    # computer a word vector
    #result = model[u'张军']
    #print(result)
    #print(len(result))

def load(filename):
    f = open(filename)
    result = dict()
    for line in f:
        line = line.strip().split('\t')
        if line[1] not in result:
            result[line[1]] = line[-1]
    return result

def similar(filename, savefile):
    model = Word2Vec.load('./cid.model')
    words = load(filename)
    f = open(savefile, 'w')
    for word in words:
        try:
            result = model.most_similar(word)
        except Exception as e:
            result = []
            pass
        if len(result)==0:
            f.write(word+'\t'+'不在w2v'+'\t'+'不在w2v'+'\n')
            continue
        top5 = list()
        for w,s in result[:5]:
            top5.append(w+':'+words[w]+':'+str(s))
        f.write(word+':'+words[word]+'\t'+','.join(top5)+'\n') 
        
 
def test_word(model_path):
    model = Word2Vec.load(model_path)
    for x in ['香梨', '熟肉','老鼠贴','汗巾']:
        print(x)
        result = model.most_similar(x)
        print(json.dumps(result, ensure_ascii=False))
        
    result = model.most_similar('梨子')
    print('梨子: \t'+json.dumps(result, ensure_ascii=False))
    result = model.most_similar('笋子')
    print('笋子: \t'+json.dumps(result, ensure_ascii=False))
    result = model.most_similar('鸭脚')
    print("鸭脚: \t"+json.dumps(result, ensure_ascii=False))
    result = model.most_similar('馍干')
    print("馍干: \t"+json.dumps(result, ensure_ascii=False))
    result = model.most_similar('早点')
    print("早点: \t"+json.dumps(result, ensure_ascii=False))
    result = model.most_similar('糖糕')
    print("糖糕: \t"+json.dumps(result, ensure_ascii=False))


if __name__=='__main__':
    #train()
    #filename, savefile = sys.argv[1:]
    test_word('./aug_jxpp.model')
    #similar(filename, savefile)

