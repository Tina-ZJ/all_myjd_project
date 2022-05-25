# -*- coding:utf8 -*-
import sys
import xlwt

def count(file1):
    f = open(file1)
    workbook = xlwt.Workbook(encoding='utf-8',style_compression=0)
    booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)

    row = 0
    header=[ u'cid3:类目名称', u'top5最相关的类目']
    for col in range(len(header)):
        booksheet.write(row, col, header[col])
    for line in f:
        row +=1
        line = line.strip()
        content=line.split('\t')
        for col in range(len(content)):
            booksheet.write(row,col,content[col])
    workbook.save('cid3_simi.xls')
         
if __name__=='__main__':
    file1 = sys.argv[1]
    count(file1)

