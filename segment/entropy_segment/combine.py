# -*- coding: utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import math
import collections
import time
import fileinput



def term_left_right_infor(left_infor, right_infor, save_file):
    term_left = collections.OrderedDict()
    term_right = collections.OrderedDict()
    f = open(save_file,'w')
    for line in fileinput.input(left_infor):
        k,v = line.strip().decode('utf8').split('\t')
        term_left[k]=v
    for line in fileinput.input(right_infor):
        k, v = line.strip().decode('utf8').split('\t')
        term_right[k]=v
    lterms = set(term_left.keys())
    rterms = set(term_right.keys())
    same_terms = lterms & rterms
    diff_left = lterms - same_terms
    diff_right = rterms - same_terms
    for term in same_terms:
        f.write(term+'\t'+str(term_left[term]) +'\t' +  str(term_right[term]) + '\n')
    for term in diff_left:
        f.write(term +'\t' + str(term_left[term]) +'\t' + '0.0' + '\n')
    for term in diff_right:
        f.write(term + '\t' + '0.0' + '\t' + str(term_right[term]) + '\n')
    f.close()

            

if __name__=='__main__':
    if len(sys.argv) !=4:
        print "input %s left_infor_file, right_infor_file, save_file_name" % (sys.argv[0])
        sys.exit()
    left_infor = sys.argv[1]
    right_infor = sys.argv[2]
    save_file = sys.argv[3]
    term_left_right_infor(left_infor,right_infor, save_file)
