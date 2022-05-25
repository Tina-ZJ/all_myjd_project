# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import re

def check_chinese(string):
    for ch in string.decode("utf8"):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def check_digit(string):
    for ch in string:
        if ch.isdigit():
            return True
    return False

def check_english(string):
    for ch in string:
        if ch.isalpha():
            return True
    return False

def check_N_V(string):
    string = string.decode('utf8')
    gz = u'[男|女]'
    gz2 = u'[春|夏|秋|冬|粗|细|厚|薄|新|大|小|长|短|季|白|皮]'
    rule = re.compile(gz)
    rule2 = re.compile(gz2)
    count = rule.findall(string)
    count2 = rule2.findall(string)
    if len(count)>0 and len(count2)>0:
        return True
    return False

def check_unit(string):
    gz = u'\d+(寸|匹|千克|公斤|升|米|厘米|分米|斤|岁|元|个月|月|磅|毫升|毫米|英尺|英寸|年|日|分|秒|片|克拉)'
    
    rule = re.compile(gz)
    string = string.decode('utf8')
    count = rule.findall(string)
    if len(count) > 0 :
        return True
    else :
        return False

def check_(string):
    gz = r'^\d*[\/-]\d*$'
    rule = re.compile(gz)
    count = rule.findall(string)
    if len(count) > 0:
        return True
    else:
        return False
def check_bad(string):
    gz = u'[女|男][春|夏|秋|冬|长|麻]'
    gz2 = u'[春夏秋冬女男]百搭'
    gz3 = u'[克][男|女]'
    rule2 = re.compile(gz2)
    rule = re.compile(gz)
    rule3 = re.compile(gz)
    count = rule.findall(string.decode("utf8"))
    count2 = rule2.findall(string.decode("utf8"))
    count3 = rule3.findall(string.decode("utf8"))
    if len(count) > 0 or len(count2) >0 or len(count3) > 0:
        return True
    else:
        return False


def filt(filename):
    f2 = open(filename,'w')
    f = open('bad','w')
    for line in sys.stdin:
        term = line.strip().split("\t")
        if len(term)!=3:
            print line.strip().decode("utf8")
        if len(term[0].decode("utf8"))>=4 and check_chinese(term[0]) and (check_digit(term[0]) or check_english(term[0])) and u'满' not in term[0] and u'减' not in term[0] and u'.' not in term[0] and u'·' not in term[0] and u'—' not in term[0] and u'_' not in term[0]:
            f.write(line.strip().decode('utf8') +'\n')
        elif check_unit(term[0].decode("utf8")) and u'分袖' not in term[0].decode("utf8") and u'分裤' not in term[0].decode("utf8") or term[0]=='@#$':
            f.write(line.strip().decode('utf8') +'\n')
        elif ('vivo' in term[0] and 'vivo'!= term[0] )or ('oppo' in term[0] and 'oppo' != term[0]) or ('iphone' in term[0] and 'iphone'!= term[0]) or ('note' in term[0] and 'note' !=term[0]) or ('mate10' in term[0] and 'mate10'!=term[0]) or (term[0].startwith('ppoa') or (term[0].startwith('ppor')) or  check_(term[0]):
            f.write(line.strip().decode('utf8')+'\n')
        elif (u'360q' in term[0].decode('utf8')):
            f.write(line.strip().decode('utf8'))
        elif check_bad(term[0]):
            f.write(line.strip().decode("utf8")+'\n')
        elif check_N_V(term[0]):
            f.write(line.strip().decode("utf8")+'\n')
        elif (u'男' in term[0] or u'女' in term[0] or u'春' in term[0] or u'夏' in term[0] or u'秋' in term[0] or u'冬' in term[0]) and (term[1]==0.0 and term[2]==0.0):
            f.write(line.strip().decode("utf8")+'\n')
        else:
            f2.write(line.strip().decode('utf8') + '\n')
    f2.close()
    f.close()

if __name__=='__main__':
    filename = sys.argv[1]
    filt(filename)
