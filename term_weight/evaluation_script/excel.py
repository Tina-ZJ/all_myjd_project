# -*- coding:utf8 -*-
import sys
import xlwt

def count(file1):
    f = open(file1)
    workbook = xlwt.Workbook(encoding='utf-8',style_compression=0)
    booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)

    row = 0
    header=[ u'sku_id', u'sku_name', u'cid3_name', u'等级打标']
    for col in range(len(header)):
        booksheet.write(row, col, header[col])
    for line in f:
        row +=1
        line = line.strip()
        content=line.split('\t')
        for col in range(len(content)):
            if col==3:
                content[col] = ','.join(content[col].split())
            booksheet.write(row,col,content[col])
    workbook.save('sku_term_weight_v2.xls')
         
if __name__=='__main__':
    file1 = sys.argv[1]
    count(file1)

