# coding:utf-8
import sys

reload(sys)
sys.setdefaultencoding('utf8')

def parseTag(label_ids, id2label, txt):
    result = []
    preLabel = ""
    words = []
    for i, id in enumerate(label_ids):
        label = id2label[id]
        if label == "O":
            if preLabel != "O" and len(words) > 0:
                token = "".join(words)
                if len(preLabel) > 0:
                    token = token
                result.append(token)
                words = []
            words.append(txt[i])
        else:
            if preLabel == "O" and len(words) > 0:
                result.append("".join(words))
                words = []
            words.append(txt[i])
            bies = label[0]
            tag = label[2:]
            if bies == "S" or bies == "E":
                token = "".join(words)
                if len(tag) > 0:
                    token = token
                result.append(token)
                words = []
    if len(words) > 0:
        token = "".join(words)
        if len(tag) > 0:
            token = token
        result.append(token)

    return result
