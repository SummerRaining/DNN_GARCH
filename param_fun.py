# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:45:17 2019

@author: a
"""


import keras,os,pickle
from keras.layers import LSTM,Input,Dropout,Dense,GRU,BatchNormalization,Add,Conv1D,MaxPool1D,GlobalAveragePooling1D
from keras import Model,Sequential
from keras.layers import Layer
from keras import backend as K
from keras import optimizers,initializers
from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import het_arch
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product


def MLE_loss(y_true, y_pred):
    #认为模型预测的是方差的log值，这样就避免了模型无法预测值有负数的问题。
    epsilon = 1e-6
    y_pred = K.exp(y_pred)
    return K.mean( 0.5*K.log(y_pred+epsilon)+y_true*y_true/(2.0*y_pred+epsilon))

def build_fcnn_model(structure = [16,8,1],activate_func = "linear",input_dim = 11):
    '''
    产生fcnn模型.
    输入input_dim(int):代表输入向量总维度。
    structure(list):代表每一层的维度。
    '''
    model = Sequential()
    model.add(Dense(units=structure[0], activation=activate_func,input_dim = input_dim))
    for notes in structure[1:-1]:
        model.add(Dense(units=notes, activation=activate_func))
    model.add(Dense(units=1, activation='linear'))
    adam = optimizers.Adam(lr=0.001, clipvalue=0.05)
    model.compile(loss=MLE_loss, optimizer=adam)
    return model

#数据预处理
def generate_data(x,p = 10,q = 5,t = 10):
    '''
    使用序列x生成多个特征的滞后项，包括
    outputs：
        x_data(array): 特征数据，大小为(num-2*length)*length，每个元素代表过去10天的残差值。
        x_var(array): 特征数据，大小为(num-2*length)*length；
                    x_var[i]代表第i日到第i-length日所计算的样本方差的对数值。
        y_data(array)：标签数据，大小为num-length，每个元素是当前的残差值。
    '''
    num = len(x)
    x_data = []
    y_data = []
    x_var = []
    
    i_begin = max(q+t-p-1,0)
    i_end = num-p-1
    
    def time_var(i,t,x):
        #包含i，往前推t日的样本方差。对数值。
        return np.log(np.var(x[i-t+1:i+1]))
    
    for i in range(i_begin,i_end+1):
        y_data.append(x[i+p])
        x_data.append(x[i:i+p])
        
        #计算方差
        tmpv = np.zeros(q)
        for j in range(1,q+1):
            tmpv[q-j] = time_var(i+p-j,t,x)
        x_var.append(tmpv)
    x_data,x_var,y_data = np.array(x_data),np.array(x_var),np.array(y_data)
    print("产生样本数{}".format(len(x_data)))
    return x_data,x_var,y_data

def train_model(x,lag_value,lag_var,var_times, structure, activation):
    #生成样本
    x_data,x_var,y_data = generate_data(x,p = lag_value,q = lag_var,t = var_times)
    x_data = np.concatenate((x_data,x_var),axis = -1)
    model = build_fcnn_model(structure = structure,activate_func = activation,input_dim=x_data.shape[-1])    

    #训练模型，将所有模型放入文件夹中。
    path = "models/temporary_fcnn_model"
    os.mkdir(path)
    checkpoint_path = os.path.join(path,'epoch_{epoch:03d}loss{loss:.3f}.h5')
    checkpoint = ModelCheckpoint(checkpoint_path,monitor='loss',\
                         save_best_only=True,save_weights_only=False,verbose=0)
    hist = model.fit(x =x_data,y=y_data,batch_size = 32,epochs = 100,\
                     verbose = 0,callbacks=[checkpoint],shuffle = True)        
    #加载最优模型，测试结果。
    model_name = sorted(os.listdir(path))[-1]
    model = keras.models.load_model(os.path.join(path,model_name),\
                                custom_objects={'MLE_loss': MLE_loss})
        

    #计算标准残差序列，对其做arch检验。
    log_sigma = model.predict(x_data)
    
    sigma = np.sqrt(np.exp(log_sigma))
    sigma = np.reshape(sigma,(-1))
    std_res = y_data/sigma
    arch_test = het_arch(std_res)
    
    #输出结果
    header = "lag_value,lag_var,var_times, structure, activation,loss,statistic,p_value"
    result = ",".join([str(x) for x in [lag_value,lag_var,var_times,\
                                        structure, activation,\
                                        np.min(hist.history['loss']),\
                                        arch_test[0],arch_test[1] ]])    
        
    #放在intermediate下面
    log_name = "intermediate/fcnn_log.csv"
    if os.path.exists(log_name):
        #读出后关闭文件。
        with open(log_name,'r') as f:
            content = f.read()
        content = content + "\n"+ result
    else:
        content = header+"\n"+result
    with open(log_name,'w') as f:
        f.write(content)
        
    #删除模型文件和对应的文件夹。
    for x in os.listdir(path):
        os.remove(os.path.join(path,x))
    os.removedirs(path)
    
def tune_parameter(x,kwargs):
    '''
    输入：x(array)，一维的时间序列。代表残差序列
    kwargs(dict):key是参数名，value是参数对应的可能组合
    '''
    #使用product函数得到list的所有组合，返回迭代器，迭代器的每一个元素是一个元组，按顺序包含参数值。
    iterations = list(product(kwargs['lag_value'],kwargs['lag_var'],kwargs['var_times'],\
            kwargs['structure'],kwargs['activation']))
    print("一共迭代{}次".format(len(iterations)))
    for i,option in tqdm(enumerate(iterations)):
        lag_value = option[0]
        lag_var = option[1]
        var_times = option[2]
        structure = option[3]
        activation = option[4]
        print("第{}轮训练".format(i))
        train_model(x,lag_value,lag_var,var_times, structure, activation)

        
if __name__ == "__main__":
    data = pd.read_csv("intermediate/arma_residual.csv")
    data.columns = ["residual"]
    arch_test = het_arch(data['residual'].values)
    print("残差数据，arch检验的统计量：{:.3f}，对应p值为:{:.4f}".format(arch_test[0],arch_test[1]))
    
    #滞后值10阶，方差一阶，10日
    # x_data,x_var,y_data = generate_data(data['residual'],p = 10,q = 5,t = 10)
    
    parameters = {"lag_value":[5,10,15],"lag_var":[1,5,10],"var_times":[5,10],\
                  "structure":[[64,32,8,1],[32,8,1],[8,1]],"activation":["relu","linear"]}
    tune_parameter(data['residual'].values,parameters)
    
# =============================================================================
#     path = "models/temporary_fcnn_model"
#     model_name = sorted(os.listdir(path))[-1]
#     model = keras.models.load_model(os.path.join(path,model_name),\
#                                 custom_objects={'MLE_loss': MLE_loss})
# =============================================================================
