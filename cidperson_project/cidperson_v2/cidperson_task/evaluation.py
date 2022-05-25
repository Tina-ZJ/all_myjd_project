# -*-coding:utf8 -*-

import sys



def acc(filename, savefile):
    f = open(filename)
    f2 = open(savefile, 'w')
    count = 0.0
    all_count = 0.0
    count_2 = 0.0
    count_3 = 0.0
    all_cid3 = set()
    for line in f:
        flag = 0
        all_count+=1
        terms = line.strip().split('\t')
        truth = terms[-3]
        predict = terms[-1].split(',')
        top1 = 0
        for i, x in enumerate(predict):
            c,n,s = x.split(':')
            if i==0:
                top1 = c
            if c == truth and i==0:
                count+=1
                flag = 1
            if c == truth and i==1:
                count_2+=1
            if c == truth and i==2:
                count_3+=1
            #if float(s) >=0.05:
            #   all_cid3.add(c+':'+n) 
        f2.write(str(flag)+'\t'+line)

    print("top1准确率为: %s" % str(count/all_count)) 
    print("top2准确率为: %s" % str((count+count_2)/all_count))
    print("top3准确率为: %s" % str((count+count_2+count_3)/all_count))
    #print(all_cid3)

if __name__=='__main__':
    filename, savefile = sys.argv[1:]
    acc(filename, savefile) 
