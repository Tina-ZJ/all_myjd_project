def parseTag(tags, line, id2label):
    res = []
    tmp = []
    for i in range(len(tags)):
        tag = id2label[tags[i]]
        bies = tag[0]
        t = tag[2:]
        tmp.append(line[i])
        if bies == "S" or bies == "E":
            res.append("".join(tmp) if len(t) == 0 else "".join(tmp) + "/" + t)
            tmp = []

    if len(tmp) > 0:
        res.append("".join(tmp) if len(t) == 0 else "".join(tmp) + "/" + t)
    return " ".join(res)


def parseTag_Ner(tags, line, id2label):
    res = []
    tmp = []
    for i in range(len(tags)):
        tag = id2label[tags[i]]
        bies = tag[0]
        t = tag[2:]
        if bies == "B":
            if len(tmp) > 0:
                res.append("".join(tmp))
                tmp = []
        tmp.append(line[i])
        if bies == "S" or bies == "E":
            res.append("".join(tmp) if len(t) == 0 else "".join(tmp) + "/" + t)
            tmp = []

    if len(tmp) > 0:
        res.append("".join(tmp) if len(t) == 0 else "".join(tmp) + "/" + t)
    return " ".join(res)
