# -*- coding:utf8 -*-
import sys




def diff(filename, filename2, savefile):
    f = open(filename)
    f1 = open(filename2)
    f2 = open(savefile,'w')
    for line in f:
        terms = line.strip().split('\t')
        terms2 = f1.readline().split('\t')
        cid1 = list()
        cid2 = list()
        for x in terms[-1].split(','):
            cid1.append(':'.join(x.split(':')[:-1]))
            break 
        
        for x in terms2[-1].split(','):
            cid2.append(':'.join(x.split(':')[:-1]))
            break
 
        if cid1!=cid2:
            f2.write(terms[0]+'\t'+cid1[0]+'\t'+cid2[0]+'\n')

    



if __name__=='__main__':
    filename, filename2, savefile = sys.argv[1:]
    diff(filename, filename2, savefile) 
