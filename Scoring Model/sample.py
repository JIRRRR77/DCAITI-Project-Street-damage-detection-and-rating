# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 20:11:00 2022

@author: YY
"""
import shutil
import csv
import pandas as pd
import os
with open('selection1.csv','r') as f:
    reader = csv.reader(f)
    column = [row[5] for row in reader]
    print(column)
# x=column[1].split('.')    
for i in range(1,490):
    x=column[i].split('\\') 
    y=x[0]+"\\"+x[1]+"\\"+x[2]+"\\"+x[2]+"\\"+x[2]+'_'+x[3]  
    #y="\\"+x[2]+'_'+x[3]  
    a= "C:\\Users\\YY\\Desktop\\"
    xx=a+y
    target_path = os.path.abspath(r'C:\Users\YY\Desktop\新建文件夹 (2)') 
    shutil.copy(xx, target_path)