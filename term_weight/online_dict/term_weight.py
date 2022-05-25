# -*- coding: utf-8 -*-  
import sys
from qpdl_utils import *

class TermWeight:
    def __init__(self, value_ori=0.001, value_nor=0.05, portion=0.5):
        self.twdict = {}
        self.value_ori = value_ori
        self.value_nor = value_nor
        self.portion = portion

    def init(self, path='termweight.txt', env='local'):
        if env == 'spark':
            ret, path = get_model_path_in_spark('termweight')
            if ret != 'OK': return path

        for l in open(path):
            if sys.version_info.major == 2 and isinstance(l, str):
                l = l.decode('utf-8')
            a = l.strip().split('\t')
            if len(a) != 2:
                logw('ERROR load TermWeight line %s'%(l))
                continue
            term= a[0]
            self.twdict[term] = {}
            for cw in a[1].split(','):
                try:
                    c,w = cw.split(':')
                    c = int(c)
                    w = float(w)
                    if c <= 0 :
                        logw('ERROR load TermWeight line %s'%(l))
                        break
                    #if abs(w) < 0.000001:
                    #    w = 0.0
                    self.twdict[term][c] = w
                except:
                    logw('parse errror %s'%(l))
                    break
        return 'OK'

    def grade(self, wl, nwl):
        
        gl = list()
        threshold = min(1, (1.0/len(wl))*(1+self.portion))
         
        for w, nw in zip(wl, nwl):
            g = 0
            if nw > threshold:
                if w >self.value_ori:
                    g = 5
                else:
                    g = 3
            else:
                if nw >self.value_nor:
                    g = 1

            gl.append(g)

        return gl 
         
    def predict(self, cid, terms):
        r = {}
        if len(terms) == 0:
            logw('terms len %d'%(len(terms)))
            return r
        else:
            r['term_weight'] = [1.0/len(terms)] * len(terms)

        if not isinstance(cid, int):
            try:
                cid = int(cid)
            except:
                logw('unknow cid %s'%cid)
                return r
                
        weight_total = 0.0
        wl, nwl = [], []
        for t in terms:
            if sys.version_info.major == 2 and isinstance(t, str):
                t = t.decode('utf-8')
            w = 0.0
            #if t not in self.twdict:
            #    logi('%s not in twdict'%t)
            #elif cid not in self.twdict[t]:
            #    logi('%d not in twdict[%s]'%(cid, t))
                
            if t in self.twdict and cid in self.twdict[t]:
                w = self.twdict[t][cid]
            wl.append(w)
            weight_total += w

        if abs(weight_total) < 0.000001 :
            #logw('weight_total < 0.000001')
            for w in wl:
                nwl.append(1.0/len(terms))
        else:
            # normalization
            for w in wl:
                nwl.append(w/weight_total)
        r = {}
        r['term_weight'] = nwl
        r['term_weight_ori'] = wl


        # set grade
        gl = self.grade(wl, nwl)

        r['grade'] = gl
         
        return r

def main():
    tw = TermWeight(value_ori=0.001, value_nor=0.05, portion=0.5)
    tw.init(sys.argv[1])
    filename = sys.argv[2]
    savefile = sys.argv[3]
    f = open(filename)
    sf = open(savefile, 'w')
    for line in f:
        fileds = line.strip().split('\t')
        if len(fileds)!=3:
            continue
        terms = fileds[1].split(',')
        cid = fileds[-1]
        
        ret = tw.predict(cid, terms)
        weight = ret['term_weight']
        gl = ret['grade']
        ori_weight = ret['term_weight_ori']
        combine = [t+':'+str(w)+':'+str(ow)+':'+str(g) for t,w,ow,g in zip(terms, weight, ori_weight,gl)]
        sf.write(line.strip()+'\t'+','.join(combine)+'\n')
    

if __name__ == "__main__":
    main()
        
                

            
