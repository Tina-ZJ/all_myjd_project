import sys
import numpy as np





def pairs(score_sorted, flag=True):
    n = len(score_sorted)
    all_pairs = set()
    for i in range(n-1):
        for j in range(i+1,n):
            if flag:
                if score_sorted[i][1] < score_sorted[j][1]:
                    all_pairs.add(score_sorted[i][0]+':'+score_sorted[j][0])
            else:
                if score_sorted[i][1] >= score_sorted[j][1]:
                    all_pairs.add(score_sorted[i][0]+':'+score_sorted[j][0])
    return all_pairs 

def predict_pairs(result):
    terms = result.split(',')
    scores = dict()
    for x in terms:
        ts = x.split(':')
        if ts[-1].strip()=='':
            print(result)
            continue
        scores[ts[0]] = float(ts[-1])
    
    score_sorted = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    predict_result = pairs(score_sorted, flag=False)
    return predict_result  
          
def accuracy(filename, filed=25):
    f = open(filename)
    right_pairs = [0.0] * filed
    all_pairs = 0.0
    for line in f:
        terms = line.strip().split('\t')
        tags = terms[3].split(';')
        score = dict()
        for tag in tags:
            t = tag.split('/')
            if len(t)!=2 or t[-1].strip()=='':
                print(line)
                continue
            score[t[0]] = int(t[1])
        score_sorted = sorted(score.items(), key=lambda x: x[1])
        truth_pairs = pairs(score_sorted)
        all_pairs+=len(truth_pairs)
        for i in range(filed):
            predict_result = predict_pairs(terms[i+4])
            same = truth_pairs & predict_result
            right_pairs[i]+=len(same)
    acc_list = [str(x/all_pairs) for x in right_pairs]  
    print("准确率分别为：")
    print('\t'.join(acc_list))



if __name__=='__main__':
    filename = sys.argv[1]
    accuracy(filename)
