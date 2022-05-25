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
    #model = Word2Vec(LineSentence(inp), size=300, window=3, min_count=2, workers=multiprocessing.cpu_count())
    model = Word2Vec(LineSentence(inp), size=128, window=3, min_count=2, workers=multiprocessing.cpu_count())
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

def test_word(model_path):
    model = Word2Vec.load(model_path)
    result = model.most_similar('标价机')
    print(json.dumps(result, ensure_ascii=False))
    result = model.most_similar('日历')
    print(json.dumps(result, ensure_ascii=False))
    result = model.most_similar('米尺')
    print(json.dumps(result, ensure_ascii=False))


if __name__=='__main__':
    train()
    #test_word('6.15_model')

