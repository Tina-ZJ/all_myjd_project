# -*- coding: utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import math
import collections
import time
import fileinput

def term():
    for line in sys.stdin:
        line_list = line.strip().decode('utf8').split()
        if len(line_list)==1:
            line_list.append(u"@#$")
        for j in range(len(line_list)-1):
            print line_list[j]+'\t'+line_list[j+1]


if __name__=='__main__':
    term()
