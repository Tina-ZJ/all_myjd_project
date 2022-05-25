# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf8")
import re


def check_english_digit(string):
    rule1 = re.compile(u'\d+')
    rule2 = re.compile(u'[a-zA-Z]+')
    count = rule1.findall(string)
    count2 = rule2.findall(string)
    if len(count) >0 or len(count2) >0:
        return True
    else:
        return False
    

def filter(filename):
    f2 = open(filename, 'w')
    f = open('bad2' ,'w')
    for line in sys.stdin:
        term = line.strip().split("\t")
        if len(term)!=3:
            print line
        if ('vivo' in term[0] and 'vivo' != term[0]) or (term[0].endswith('plus') and term[0]!= 'plus') or ('oppo' in term[0] and 'oppo' !=term[0]) or ('iphone' in term[0] and 'iphone' !=term[0]) or ('note' in term[0] and 'note' !=term[0] and 'notebook' !=term[0] ) or (term[0].startswith('oppa')) or (term[0].startswith('ppor')) or (term[0]=='p20pro'):
            f.write(line.strip().decode('utf8')+'\n')
        else:
            f2.write(line.strip().decode('utf8') +'\n')
    f2.close()
    f.close()

def filter2(filename):
    f2 = open(filename, 'w')
    f = open('bad3','w')
    for line in sys.stdin:
        term = line.strip().decode('utf8').split('\t')
        if (u'荣耀' in term[0] or u'畅享' in term[0] or u'分期' in term[0] or u'魅蓝' in term[0] or u'ipad' in term[0]) and check_english_digit(term[0]):
            f.write(line.strip().decode('utf8')+'\n')
        else:
            f2.write(line.strip().decode('utf8')+'\n')

if __name__=='__main__':
    file1 = sys.argv[1]
    filter2(file1)
