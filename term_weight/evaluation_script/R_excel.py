# -*-coding: utf8 -*-
import xlrd
import sys

def read(filename, savefile):
    f = open(savefile,'w')
    excel = xlrd.open_workbook(filename)
    sheet = excel.sheet_by_index(0)
    for i in range(1, sheet.nrows):
        temp = list()
        for j in range(sheet.ncols):
            t = sheet.cell(i,j).value
            temp.append(t)
        f.write('\t'.join(temp)+'\n')

if __name__=='__main__':
    filename = sys.argv[1]
    savefile = sys.argv[2]
    read(filename, savefile)
 
            
