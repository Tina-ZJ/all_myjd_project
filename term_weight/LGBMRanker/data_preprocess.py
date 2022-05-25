# -*-coding:utf8 -*-
import pandas as pd
import sys



def preprocess(data):
    data = data.replace('[','').replace(']','')
    if len(data)==0:
        return ''
    t = [float(x) for x in data.split(',')]
    if len(t)!=10:
        return ''
    t[0] = int(t[0]) 
    t[1] = int(t[1]) 
    return t

def txt2csv(filename, savefile, groupfile):
    f = open(filename)
    result = list()
    group_nums = list()
    group = ''
    b = 0
    for i, line in enumerate(f):
        terms = line.split('\t')
        # record query
        if len(terms)!=8:
            print(line)
            continue
        data = preprocess(terms[6])
        if data=='':
            print(line)
            continue
        label = int(terms[-1])
        feature =[label]+ [terms[0]]+[terms[5]]+data
        if len(feature)!=13:
            print(line)
            continue
        q = terms[0]
        # group
        if q!=group:
            
            nums = i-b
            b = i
            group = q
            if nums>0:
                group_nums.append(nums)
         
        result.append(feature)
    last = i-b+1
    if last>0:
        group_nums.append(last)

    all_nums = sum(group_nums)
    print("all samples %d " % all_nums)

    group_nums = [str(x) for x in group_nums]
    f2 = open(groupfile,'w')
    f2.write('\n'.join(group_nums))
 
    df = pd.DataFrame(result)
    columns = ['label', 'query','term','cid3', 'brand', 'ctf', 'ctf_cofidence', 'rf', 'icf', 'icf_confidence', 'icf_max', 'igm', 'entropy']
    df.columns = columns
    df.to_csv(savefile, index=False)
 

if __name__=='__main__':
    filename, savefile, groupfile = sys.argv[1:]
    txt2csv(filename, savefile, groupfile) 
