# -*- coding: utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import math
import collections
import time
import fileinput

def calculate(term):
    dt = collections.OrderedDict()
    infor = collections.OrderedDict()
    for k, v in term.items():
        score = 0.0
        for x in v:
            sums = 0.0
            if x not in dt:
                dt[x]=0
            dt[x]+=1
        for word in dt:
            sums+=dt[word]
        for word in dt:
            p = float(dt[word]) / sums
            score+=(-p)*math.log(p)
        infor[k]=score
        dt = collections.OrderedDict()
    return infor

def info(leftfile, rightfile):
    f1 = open(rightfile,'w')
    f2 = open(leftfile, 'w')
    right = collections.OrderedDict()
    left = collections.OrderedDict()
    t0 = time.time()
    for line in sys.stdin:
        k,v = line.strip().decode('utf8').split('\t')
        if k not in right:
            right[k]=list()
        if v not in left:
            left[v]=list()
        right[k].append(v)
        left[v].append(k)
    right_infor = calculate(right)
    left_infor = calculate(left)
    for word in right_infor:
        f1.write(word.decode('utf8') + '\t' + str(right_infor[word]) +'\n')
    for word in left_infor:
        f2.write(word.decode('utf8') + '\t' + str(left_infor[word]) +'\n')


if __name__=='__main__' :
    left = sys.argv[1]
    right = sys.argv[2]
    info(left,right)
