# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf8")

def augment():
    for line in sys.stdin:
        line_list = line.strip().decode("utf8").split("\t")
        if line_list[0]=='vivo' or line_list[0]=='oppo' or line_list[0]=='iphone' or line_list[0]=='plus' or line_list[0]==u'note':
            line_list[1]= str(float(line_list[1])*5)
            line_list[2]=str(float(line_list[2])*5)
        print "\t".join(line_list)

if __name__=='__main__':
    augment()
