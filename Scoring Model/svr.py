# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:08:49 2022

@author: YY
"""
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
#from sklearn.preprocessing import StandardScaler


class SVR_Model:
    def __init__(self, x_train, y_train, optimization=True):
        """

        :param x_train: training set
        :param y_train: training label
        :param x_test: test set
        :param y_test: test label
        :param optimization: bool
        """
        self.x_train = x_train
        self.y_train = y_train
        #self.y_train = y_train.ravel()  # 将label转为一维数组,shape: (11821, 1)-->(11821,)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        self.x_train, self.y_train, random_state=42, test_size=0.2)
        #stdsc = StandardScaler()
        #self.x_train = stdsc.fit_transform(self.x_train)
        #self.x_test= stdsc.transform(self.x_test.reshape(-1, 1))
        self.model = None
        self.optimization = optimization
        
        
    def get_score(self, params):
        """
        Used as objective function, get model score

        :param params: list, Model hyperparameters
        :return: float, model error
        """
        assert self.optimization is True
        # kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        self.model = SVR(kernel='rbf',C=params[0], gamma=params[1])
        self.model.fit(self.x_train, self.y_train)  # 训练模型
       # result = self.model.predict(self.x_test) # 对测试集进行预测
        Error = mean_squared_error(self.y_test, self.model.predict(self.x_test))  # 计算测试正确率
        #print("线性核函数支持向量机的默认评估值为：", self.model.score(self.x_test, self.y_test))
        #print("线性核函数支持向量机的R_squared值为：", r2_score(self.y_test, self.model.predict(self.x_test))
        
        return Error

    def train_svr(self, params):
   # def train_svr(self):
        """
        Usually train SVM

        :param params: list, model params
        :return: None
        """
        assert self.optimization is False
        self.model = SVR(kernel='rbf',C=params[0], gamma=params[1])
        #self.model = SVR(kernel='rbf')
        self.model.fit(self.x_train, self.y_train)  # 训练模型
        #result = self.model.predict(self.x_test) # 对测试集进行预测
        R2=r2_score(self.y_test, self.model.predict(self.x_test))
        print("R_squared：",R2)#越接近1 回归模型越好r = len(x_test) + 1
        print("MSE:", mean_squared_error(self.y_test, self.model.predict(self.x_test)))
        r = len(self.x_test) + 1
        plt.plot(np.arange(1,r), self.model.predict(self.x_test), 'go-', label="predict")
        plt.plot(np.arange(1,r), self.y_test, 'co-', label="real")
        plt.legend()
        plt.show()
        #pre=self.model.predict(self.x_test)
        
        