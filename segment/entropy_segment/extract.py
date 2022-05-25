# -*- coding: utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def good_term(file1, file2):
    f1 = open(file1)
    f2 = open(file2, 'w+')
    for line in f1:
        line = line.strip().decode('utf8')
        line_list = line.split('\t')
        tag = line_list[3].split()
        if len(tag)==1 and (u'/B' in tag[0] or u'/P' in tag[0] or u'/BP' in tag[0] or u'/S' in tag[0]):
            f2.write(line+'\n')
if __name__=='__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    good_term(file1, file2)
