# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 12:55:04 2021

@author: Administrator
"""


import xlrd
import xlwt
import numpy as np
import os
from tqdm import tqdm

if __name__ == '__main__':
    data = xlrd.open_workbook('./Patient_List.xlsx')
    table = data.sheets()[0]
    List_all = table.col_values(0)[1:]
    
    Files = os.listdir('../Dataset/Image')
    Files = [File.split('.nii.gz')[0] for File in Files]

    New_List = xlwt.Workbook()
    sheet_New = New_List.add_sheet(u'sheet1', cell_overwrite_ok=True)
    sheet_New.write(0,0,table.cell_value(0,0))
    sheet_New.write(0,1,table.cell_value(0,1))
    sheet_New.write(0,2,table.cell_value(0,2))
    sheet_New.write(0,3,table.cell_value(0,3))
    sheet_New.write(0,4,table.cell_value(0,4))
    sheet_New.write(0,5,table.cell_value(0,5))
    sheet_New.write(0,6,table.cell_value(0,6))
    sheet_New.write(0,7,table.cell_value(0,7))

    num = 0
    for File in tqdm(Files):
        num = num+1
        ind = List_all.index(int(File))+1

        sheet_New.write(num,0,table.cell_value(ind,0))
        sheet_New.write(num,1,table.cell_value(ind,1))
        sheet_New.write(num,2,table.cell_value(ind,2))
        sheet_New.write(num,3,table.cell_value(ind,3))
        sheet_New.write(num,4,table.cell_value(ind,4))
        sheet_New.write(num,5,table.cell_value(ind,5))
        sheet_New.write(num,6,table.cell_value(ind,6))
        sheet_New.write(num,7,table.cell_value(ind,7))

    New_List.save('../Dataset/PatientList.xls')