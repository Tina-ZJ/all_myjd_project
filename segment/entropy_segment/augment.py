# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf8")

def augment():
    for line in sys.stdin:
        line_list = line.strip().decode("utf8").split("\t")
        word_len = len(line_list[0])
        if len(line_list) !=3:
            continue
        if word_len==1:
            word_len = 0.4
            line_list[1]=str((float(line_list[1])+1.0)*word_len)
            line_list[2]= str((float(line_list[2])+1.0)*word_len)
        elif word_len<=3:
            word_len*=(1.0+0.5*word_len)
            line_list[1] =str((float(line_list[1])+9.0)*word_len)
            line_list[2] =str((float(line_list[2])+9.0)*word_len)
        else:
            word_len*=(1.0+0.5*word_len)
            line_list[1] = str((float(line_list[1])+8.0)*word_len)
            line_list[2] = str((float(line_list[2])+8.0)*word_len)
        print "\t".join(line_list)

if __name__=='__main__':
    augment()
