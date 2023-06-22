# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:18:18 2022

@author: YY
"""
import numpy as np
from svr import SVR_Model
from sklearn.datasets import load_boston
from pso import PSO
from RF import RF_Model
import csv
boston = load_boston()
# 查看数据描述
#y_train = boston.target
#x_train = boston.data
with open('C:/Users\YY\Desktop\新建文件夹 (4)\\results(面积无置信度).csv','r') as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
#print (rows)#输出所有数据


data=np.array(rows)#rows是数据类型是‘list',转化为数组类型好处理
x_train=data[:,1:6]
y_train=data[:,6]
#params=[7.2655,0.0048]#面积无置信度zw R0.4 MSE 0.27
params=[57.34,0.0923]#面积无置信度tw R0.4935 MSE 0.4839
#params=[34.4768,0.88215]#对角线无置信度zw R0.38 MSE 0.2824
#params=[84.6729,0.9]#对角线无置信度tw R0.30387 MSE 0.6651
#params=[100,0.9]#对角线乘置信度zw R0.4753 MSE 0.239
#params=[1,0.9]#对角线乘置信度tw R0.48979 MSE 0.487487

#params=[100,0.516]#面积乘置信度tw R0.4896544 MSE 0.48762
#params=[43.7098,0.001]#面积乘置信度zw R0.40386 MSE 0.27166

svrr = SVR_Model(x_train, y_train,optimization=False)
svrr.train_svr(params)

#var_size = [[1, 100], [0.001, 0.9]]
#svr = SVR_Model(x_train, y_train,optimization=True)
#pso = PSO(svr.get_score, 2, 10, var_size)
#pso.run()


#params=[[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
#rf = RF_Model(x_train, y_train,optimization=True)
#rf.get_score(params)

#params=[4,14,9]#面积乘置信度tw R0.547822 MSE 0.43042
#params=[2,15,14]#面积乘置信度zw R0.2892606 MSE 0.32389
#params=[4,6,8]#对角线乘置信度zw R0.18225 MSE 0.372654
#params=[5,15,2]#对角线乘置信度tw R0.49935 MSE 0.47835
#params=[2,7,5]#对角线无置信度zw R0.518 MSE 0.21949

#params=[4,16,4]#对角线无置信度tw R0.5503 MSE 0.4296
params=[2,16,4]#面积无置信度tw R0.5657 MSE 0.414942

#params=[4,12,6]#面积无置信度zw R0.4435 MSE 0.253576

RFF= RF_Model(x_train, y_train,optimization=False)
RFF.train_rf(params)