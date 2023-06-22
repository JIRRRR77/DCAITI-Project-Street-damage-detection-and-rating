# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 01:55:53 2022

@author: YY
"""

import os
import shutil

source_path = os.path.abspath(r'C:\Users\YY\Desktop\Los1\Bilddaten')     #源文件夹
target_path = os.path.abspath(r'C:\Users\YY\Desktop\新建文件夹 (4)')    #目标文件夹


for root, dirs, files in os.walk(source_path):
    for file in files:
        src_file = os.path.join(root, file)
        shutil.copy(src_file, target_path)
        print(src_file)

print('复制完成')