# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:43:36 2019

@author: a
"""
from simple_model_structure import build_lstm_model,build_cnn_model,build_gru_model,build_fcnn_model,GARCH_DNN
from util import plot_four_subgraph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_arch

#数据预处理
def generate_data_lag(x,length = 10):
    '''
    使用序列x生成只包含，过去10日值与10日计算的方差。
    x_data(array):特征数据，大小为(num-length)*length，每个元素代表过去10天的残差值。
    y_data(array)：标签数据，大小为num-length，每个元素是当前的残差值。
    x_var(array):特征数据，大小为num-length,每个元素是过去10天计算的样本方差。
    '''
    sample_num = len(x)
    x_data = []
    y_data = []
    x_var = []
    for i in range(sample_num-length):
        x_data.append(x[i:i+length])
        x_var.append(np.log(np.var(x[i:i+length])))
        y_data.append(x[i+length])
        
    #转换成array。
    x_data,y_data,x_var = np.array(x_data),np.array(y_data),np.array(x_var).reshape(-1,1)
    return x_data,y_data,x_var 

#数据预处理
def generate_data_multilag(x,length = 10):
    '''
    使用序列x生成多个特征的滞后项，包括
    outputs：
        x_data(array): 特征数据，大小为(num-2*length)*length，每个元素代表过去10天的残差值。
        x_data_square(array): 特征数据，x_data的平方值。
        x_var(array): 特征数据，大小为(num-2*length)*length；
                    x_var[i]代表第i日到第i-length日所计算的样本方差的对数值。
        y_data(array)：标签数据，大小为num-length，每个元素是当前的残差值。
    '''
    sample_num = len(x)
    x_data = []
    y_data = []
    x_var = []
    for i in range(sample_num-2*length+1):
        x_data.append(x[i+length-1:i+2*length-1])
        y_data.append(x[i+2*length-1])
        #计算第i日的样本方差
        tmp_var = np.zeros(length)
        for j in range(length):
            #当i=0时，j=0时，代表计算第1个样本，前1日滞后的样本方差。根据9到19日的值计算。
            #当i=0，j=9是，代表计算第1个样本，前10日滞后的样本方差，根据0到10日的值计算。
            #当i=n-2length,j=0是，代表计算最后一个样本，前1日滞后的方差，根据 n-1-length到n-1的值计算。
            end = i+2*length-j-1
            start = i+length-j-1
            tmp_var[length-j-1] = np.log(np.var(x[start:end]))
        x_var.append(tmp_var)
        
    #转换成array。
    x_data,y_data,x_var = np.array(x_data),np.array(y_data),np.array(x_var)
    x_data_square = x_data**2
    return x_data,x_data_square,x_var,y_data 

if __name__ == "__main__":    
    data = pd.read_csv("intermediate/arma_residual.csv")
    data.columns = ["residual"]
    arch_test = het_arch(data['residual'].values)
    print("残差数据，arch检验的统计量：{:.3f}，对应p值为:{:.4f}".format(arch_test[0],arch_test[1]))
    
    #生成只包含滞后值的样本。
    x_data,y_data,x_var = generate_data_lag(data['residual'].values,length = 20) 
    x_data = np.concatenate([x_data,x_var],axis = -1)
    
    #开始训练模型.
    fcnn_model = GARCH_DNN(build_fcnn_model(x_data.shape[-1]),name="simple_FCNN_GARCH",x = x_data,y = y_data)
    fcnn_model.fit()
    fcnn_model.print_analyse()
    
# =============================================================================
#     #包含滞后值，平方项和方差的样本
#     x_data,x_data_square,x_var,y_data = generate_data_multilag(data['residual'].values) 
#     x_stack = np.stack([x_data,x_data_square,x_var],axis = -1)
#     x_flatten = x_stack.reshape([len(x_stack),-1])
#     
#     #开始训练模型.
#     fcnn_model = GARCH_DNN(build_fcnn_model(x_flatten.shape[-1]),name="Multilag_FCNN_GARCH",x = x_flatten,y = y_data)
#     fcnn_model.fit()
#     fcnn_model.print_analyse()
#     
#     lstm_model = GARCH_DNN(build_lstm_model(x_stack.shape[-1]),name="Multilag_LSTM_GARCH",\
#                            x = x_stack,y = y_data)
#     lstm_model.fit()
#     lstm_model.print_analyse()
#     
#     
#     gru_model = GARCH_DNN(build_gru_model(x_stack.shape[-1]),name="Multilag_GRU_GARCH",\
#                           x = x_stack,y = y_data)
#     gru_model.fit()
#     gru_model.print_analyse()
#     
#     cnn_model = GARCH_DNN(build_cnn_model(),name="Multilag_CNN_GARCH",\
#                           x = x_stack,y = y_data)
#     cnn_model.fit()
#     cnn_model.print_analyse()    
# 
# =============================================================================
# =============================================================================
#     #损失值随迭代次数降低的曲线
#     model_names = ["FCNN_GARCH", "LSTM_GARCH","GRU_GARCH", "CNN_GARCH"]
#     hist_loss = [nn_model.history,lstm_model.history,gru_model.history,cnn_model.history]
#     colors = ['red','cyan','black','blue']
#     shape = ['+',',',',','x']
#     
#     plot_four_subgraph(model_names,hist_loss,colors,shape)
# =============================================================================
    