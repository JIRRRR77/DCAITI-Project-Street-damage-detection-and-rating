# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 05:14:58 2022

@author: YY
"""

import os
f=28000
m=28000
s=0
for x in range(143):   
    #C:\Users\YY\Desktop\Los1\Bilddaten
    a="C:\\Users\\YY\\Desktop\\Los1\\Bilddaten"
    b= "\\" +str(f)+ "\\"+str(f)+"/"
    path=a+b


    filelist = os.listdir(path)
   
    n=0
    for i in filelist:
        old= path+filelist[n]
        new=path+str(m)+'_'+filelist[n]
        #new=path+str(n)+'.JPG'
        os.rename(old, new)
        print(old,'======>',new)
        #s+=1
        n+=1
    f+=1
    m+=1