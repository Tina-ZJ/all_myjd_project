# -*- coding: utf-8 -*-
import codecs
import numpy as np
import os
import pickle
import codecs

def get_embs(embfile,word2index_pre, index2word, vocab_size, embed_size=128):
    data = np.loadtxt(embfile,dtype=np.float32)
    pre_embs = [[]] * vocab_size
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    for i in range(vocab_size):
        if i in index2word:
            word = index2word[i]
            if word in word2index_pre:
                idx = word2index_pre[word]
                pre_embs[i] = data[idx,:]
            else:
                pre_embs[i] = np.random.uniform(-bound, bound, embed_size) 
        else: 
            pre_embs[i] = np.random.uniform(-bound, bound, embed_size) 
    word_embs = np.array(pre_embs)
    print(word_embs.shape)
    return word_embs 

def load_test_new(test_file,word2index):
    max_v = max(word2index.values())
    cls = max_v + 1
    sep = max_v + 2
    
    Xinput_ids, Xinput_mask, Xsegment_ids =[], [], []
    gender_dict = {0:'男性',1:'女性',2:'家庭',3:'性别未知'}
    age_dict = {0:'年龄未知', 1:'25岁以下',2:'26~35',3:'36~45',4:'46~55',5:'55以下'} 
    lines = []
    with open(test_file) as f:
        for line in f:
            line_list = line.strip('\r\n').split('\t')
            lines.append(line)
            terms = line_list[1].replace('[','').replace(']','').split(' ')
            terms = [int(x) for x in terms]
            genders=[int(line_list[2])+max_v+3]
            ages=[int(line_list[3])+max_v+7]
            session = line_list[4].replace('[','').replace(']','').split(' ')
            session = [int(x) for x in session][:20]
            input_ids = [cls]+terms+[sep]+session+[sep]+genders+[sep]+ages+[sep]
            Xinput_ids.append(input_ids)
            input_mask = [ 1 if x >0 else 0 for x in input_ids]
            Xinput_mask.append(input_mask)
            Xsegment_ids.append([0]*35)
    return Xinput_ids, Xinput_mask, Xsegment_ids, lines

def load_test(test_file,word2index, index2word):
    max_v = max(word2index.values())
    cls = max_v + 1
    sep = max_v + 2
   
    gender_dict = {0:'男性',1:'女性',2:'家庭',3:'性别未知'}
    age_dict = {0:'年龄未知', 1:'25岁以下',2:'26~35',3:'36~45',4:'46~55',5:'55以下'} 
    Xinput_ids, Xinput_mask, Xsegment_ids =[], [], []
    lines = []
    count = 0.0
    with open(test_file) as f:
        for line in f:
            line_list = line.strip().split('\t')
            keyword_seg = line_list[-1].split(',')
            terms = [word2index[x] if x in word2index else 1 for x in keyword_seg]
            terms = terms[:8] + (8 - len(terms))*[0]
            #terms = terms[:6]
            genders_ = line_list[1].replace('[','').replace(']','')
            genders=[int(genders_)+max_v+3]
            ages_ = line_list[2].replace('[','').replace(']','')
            ages=[int(ages_)+max_v+7]
            session = line_list[3].replace('[','').replace(']','').split(',')
            #session = [int(x) for x in session][:15]
            session = [int(x) for x in session][:20]
            # only true word
            true_word = []
            for x in session:
                if x!=0:
                    true_word.append(x)
            # session query
            session_querys = list()
            for i in range(int(len(session)/5)):
                t = session[i*5:(i+1)*5]
                t_s = []
                for x in t:
                    if x in index2word:
                        t_s.append(index2word[x]) 
                if len(t_s)>0:
                    session_querys.append(''.join(t_s))
                else:
                    break
            #session_str = [index2word[x] if x in index2word else '[PAD]' for x in session]
            if len(session_querys)==0:
                session_querys.append('没有实时session')
                count+=1
                #session=[]
            input_ids = [cls]+terms+[sep]+session+[sep]+genders+[sep]+ages+[sep]
            #input_ids = [cls]+terms+[sep]+true_word+[sep]+genders+[sep]+ages+[sep]
            Xinput_ids.append(input_ids)
            input_mask = [ 1 if x >0 else 0 for x in input_ids]
            Xinput_mask.append(input_mask)
            #segment_ids = [0]*(len(terms)+2)+[1]*(len(session)+1)+[2]*2+[3]*2
            #Xsegment_ids.append(segment_ids)
            Xsegment_ids.append([0]*len(input_ids))
            result = line_list[0]+'\t'+gender_dict[int(genders_)]+'\t'+age_dict[int(ages_)]+'\t'+';'.join(session_querys)+'\t'+line_list[-3]+'\t'+line_list[-2]
            lines.append(result)
    print(count) 
    return Xinput_ids, Xinput_mask, Xsegment_ids, lines

def load_test_v2(test_file,word2index, index2word):
    max_v = max(word2index.values())
    cls = max_v + 1
    sep = max_v + 2
   
    gender_dict = {0:'男性',1:'女性',2:'家庭',3:'性别未知'}
    age_dict = {0:'年龄未知', 1:'25岁以下',2:'26~35',3:'36~45',4:'46~55',5:'55以下'} 
    Xinput_ids, Xinput_mask, Xsegment_ids =[], [], []
    lines = []
    count = 0.0
    with open(test_file) as f:
        for line in f:
            line_list = line.strip().split('\t')
            keyword_seg = line_list[-1].split(',')
            terms = [word2index[x] if x in word2index else 1 for x in keyword_seg]
            terms = terms[:8] + (8 - len(terms))*[0]
           
            #terms = terms[:6]
            genders_ = line_list[1].replace('[','').replace(']','')
            genders=[int(genders_)+max_v+3]
            ages_ = line_list[5].replace('[','').replace(']','')
            ages=[int(ages_)+max_v+7]
            session = line_list[6].replace('[','').replace(']','').split(',')
            #session = [int(x) for x in session][:15]
            session = [int(x) for x in session][:20]
            session_ = []
            for x in session[:20]:
                if int(x)!=0:
                    session_.append(int(x))
            cids_realt = line_list[2].replace('[','').replace(']','').split(',')
            # get id
            cids_realt_ =[]
            for x in cids_realt:
                if int(x)!=0:
                    t = '__cid__'+str(x)
                    if t in word2index:
                        cids_realt_.append(word2index[t])
                    else: 
                        cids_realt_.append(word2index.get('__cid__new',0))
            # only true word
            true_word = []
            for x in session:
                if x!=0:
                    true_word.append(x)
            # session query
            session_querys = list()
            for i in range(int(len(session)/5)):
                t = session[i*5:(i+1)*5]
                t_s = []
                for x in t:
                    if x in index2word:
                        t_s.append(index2word[x]) 
                if len(t_s)>0:
                    session_querys.append(''.join(t_s))
                else:
                    break
            #session_str = [index2word[x] if x in index2word else '[PAD]' for x in session]
            if len(session_querys)==0:
                session_querys.append('没有实时session')
                count+=1
                #session=[]
            #input_ids = [cls]+genders+[sep]+ages+[sep]+cids_realt_+[sep]+terms+[sep]+session_+[sep]
            #input_ids = [cls]+genders+[sep]+ages+[sep]+terms+[sep]+session_+[sep]
            input_ids = [cls]+terms+[sep]+session+[sep]+genders+[sep]+ages+[sep]
            #input_ids = [cls]+terms+[sep]+[0]*20+[sep]+genders+[sep]+ages+[sep]
            Xinput_ids.append(input_ids)
            input_mask = [ 1 if x >0 else 0 for x in input_ids]
            Xinput_mask.append(input_mask)
            #segment_ids = [0]*(len(terms)+2)+[1]*(len(session)+1)+[2]*2+[3]*2
            #Xsegment_ids.append(segment_ids)
            Xsegment_ids.append([0]*len(input_ids))
            result = line_list[0]+'\t'+gender_dict[int(genders_)]+'\t'+age_dict[int(ages_)]+'\t'+';'.join(session_querys)+'\t'+line_list[2]+'\t'+line_list[-4]+'\t'+line_list[-3]
            lines.append(result)
    print(count) 
    return Xinput_ids, Xinput_mask, Xsegment_ids, lines

def load_test_v3(test_file,word2index, index2word):
    #max_v = max(word2index.values())
    #cls = max_v + 1
    #sep = max_v + 2
   
    gender_dict = {0:'男性',1:'女性',2:'家庭',3:'性别未知'}
    age_dict = {0:'年龄未知', 1:'25岁以下',2:'26~35',3:'36~45',4:'46~55',5:'55以下'} 
    Xinput_ids, Xinput_mask, Xsegment_ids =[], [], []
    lines = []
    count = 0.0
    with open(test_file) as f:
        for line in f:
            line_list = line.strip().split('\t')
            keyword_seg = line_list[-1].split(',')
            terms = [word2index[x] if x in word2index else 1 for x in keyword_seg]
            query_input_ids = terms[:8] + (8 - len(terms))*[0]
          
             
            genders_ = line_list[1].replace('[','').replace(']','')
            genders = '__gender__'+str(genders_)
            genders=[word2index.get(genders,1)]
            
            ages_ = line_list[5].replace('[','').replace(']','')
            ages = '__age__'+str(ages_)
            ages = [word2index.get(ages,1)]
            
            cids_realt = line_list[2].replace('[','').replace(']','').split(',')
            cids_realt = [word2index.get('__cid__'+str(x),1) if int(x)>0 else 0 for x in cids_realt]
            
             
            brands_realt = line_list[3].replace('[','').replace(']','').split(',')
            brands_realt = [word2index.get('__brand__'+str(x),1) if int(x)>0 else 0 for x in brands_realt]


            user_input_ids = genders + ages + cids_realt + brands_realt

            session = line_list[6].replace('[','').replace(']','').split(',')
            #session = [int(x) for x in session][:15]
            session_input_ids = [int(x) for x in session][:20]
            session_ = []
            for x in session[:20]:
                if int(x)!=0:
                    session_.append(int(x))
           

            # only true word
            true_word = []
            for x in session:
                if x!=0:
                    true_word.append(x)
            # session query
            session_querys = list()
            for i in range(int(len(session_input_ids)/5)):
                t = session_input_ids[i*5:(i+1)*5]
                t_s = []
                for x in t:
                    if x in index2word:
                        t_s.append(index2word[x]) 
                if len(t_s)>0:
                    session_querys.append(''.join(t_s))
                else:
                    break
            #session_str = [index2word[x] if x in index2word else '[PAD]' for x in session]
            if len(session_querys)==0:
                session_querys.append('没有实时session')
                count+=1
                #session=[]
            #input_ids = [cls]+genders+[sep]+ages+[sep]+cids_realt_+[sep]+terms+[sep]+session_+[sep]
            #input_ids = [cls]+genders+[sep]+ages+[sep]+terms+[sep]+session_+[sep]
            #input_ids = [cls]+terms+[sep]+session+[sep]+genders+[sep]+ages+[sep]
            #input_ids = [cls]+terms+[sep]+[0]*20+[sep]+genders+[sep]+ages+[sep]
            input_ids = user_input_ids + session_input_ids + query_input_ids
 
            Xinput_ids.append(input_ids)
            input_mask = [ 1 if x >0 else 0 for x in input_ids]
            Xinput_mask.append(input_mask)
            #segment_ids = [0]*(len(terms)+2)+[1]*(len(session)+1)+[2]*2+[3]*2
            #Xsegment_ids.append(segment_ids)
            Xsegment_ids.append([0]*len(input_ids))
            result = line_list[0]+'\t'+gender_dict[int(genders_)]+'\t'+age_dict[int(ages_)]+'\t'+';'.join(session_querys)+'\t'+line_list[2]+'\t'+line_list[-4]+'\t'+line_list[-3]
            lines.append(result)
    print(count) 
    return Xinput_ids, Xinput_mask, Xsegment_ids, lines
            
             
            
             
def create_term(term_file):
    term2idx = {}
    idx2term = {}
    with open(term_file ) as f:
        for line in f:
            line = line.replace('\r\n','\n')
            
             
def create_term(term_file):
    term2idx = {}
    idx2term = {}
    with open(term_file ) as f:
        for line in f:
            line = line.replace('\r\n','\n')
            line = line.strip('\n')
            if line:
                tokens = line.split('\t')
                if len(tokens) !=2:
                    print (line)
                term2idx[tokens[1]] = int(tokens[0])
                idx2term[int(tokens[0])] = tokens[1]
    return term2idx, idx2term

        
def create_label(label_file):
    label2idx = {}
    idx2label = {}
    with open(label_file) as f:
        for i, line in enumerate(f):
            line = line.replace('\r\n','\n')
            line = line.strip()
            if line:
                tokens = line.split('\t')
                label2idx[tokens[0]] = int(tokens[1])
                idx2label[int(tokens[1])] = tokens[0]
    return label2idx, idx2label


    
