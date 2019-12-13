# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 08:47:05 2019

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

def MLE_loss(y_true, y_pred):
    #认为模型预测的是方差的log值，这样就避免了模型无法预测值有负数的问题。
    epsilon = 1e-6
    y_pred = K.exp(y_pred)
    return K.mean( 0.5*K.log(y_pred+epsilon)+y_true*y_true/(2.0*y_pred+epsilon))

def build_lstm_model(input_dims = 1):    
    '''
    产生LSTM模型.
    输入input_dims(int):代表输入的通道数。
    返回lstm模型。
    '''
    inputs = Input(shape = (None,input_dims))
    x = LSTM(32,activation='tanh',return_sequences=True)(inputs)
    x = Dropout(0.5)(x)
    h = LSTM(16,activation='tanh',return_sequences=False)(x)
    y = Dense(1,use_bias=True,activation='linear')(h)
    model = Model(inputs = inputs,outputs = y)
    
    sgd = optimizers.SGD(lr=0.001, clipvalue=0.05)
    model.compile(optimizer=sgd,loss=MLE_loss)
    return model

def build_gru_model(input_dims = 1):    
    '''
    产生GRU模型.
    输入input_dims(int):代表输入的通道数。
    返回gru模型。
    '''
    #改进模型结构，将方差这一项放到最后面融合。
    inputs = Input(shape = (None,input_dims))
    x = GRU(32,activation='tanh',return_sequences=True)(inputs)
    x = Dropout(0.5)(x)
    h = GRU(8,activation='tanh',return_sequences=False)(x)
    y = Dense(1,use_bias=True,activation='linear')(h)
    model = Model(inputs = inputs,outputs = y)
    
    sgd = optimizers.SGD(lr=0.001, clipvalue=0.05)
    model.compile(optimizer=sgd,loss=MLE_loss)
    return model

def build_fcnn_model(structure = [16,8,1],activate_func = "linear",input_dim = 11):
    '''
    产生fcnn模型.
    输入input_dim(int):代表输入向量总维度。
    structure(list):代表每一层的维度。
    '''
    model = Sequential()
    for notes in structure[:-1]:
        model.add(Dense(units=notes, activation=activate_func))
    model.add(Dense(units=1, activation='linear'))
    adam = optimizers.Adam(lr=0.001, clipvalue=0.05)
    model.compile(loss=MLE_loss, optimizer=adam)
    return model

def build_cnn_model():
    '''
    cnn可以自适应输入的channel数，通道维度可以变化，不影响cnn的定义。
    '''
    model = Sequential()
    model.add(Conv1D(filters = 8,kernel_size = 3,strides = 1,padding = "same",activation = 'relu'))
    model.add(Conv1D(filters = 8,kernel_size = 3,strides = 1,padding = "same",activation = 'relu'))
    model.add(MaxPool1D(strides = 2))
    
    model.add(Conv1D(filters = 16,kernel_size = 3,strides = 1,padding = "same",activation = 'relu'))
    model.add(Conv1D(filters = 16,kernel_size = 3,strides = 1,padding = "same",activation = 'relu'))
    model.add(MaxPool1D(strides = 2))
    
    model.add(Conv1D(filters = 32,kernel_size = 3,strides = 1,padding = "same",activation = 'relu'))
    model.add(Conv1D(filters = 32,kernel_size = 3,strides = 1,padding = "same",activation = 'relu'))
    
    model.add(GlobalAveragePooling1D())
    # model.add(Reshape((3, 4), input_shape=(12,)))
    model.add(Dense(units=8, activation='tanh'))
    model.add(Dense(units=1, activation='linear'))
    
    adam = optimizers.Adam(lr=0.001, clipvalue=0.05)
    model.compile(loss=MLE_loss, optimizer=adam)
    return model

class GARCH_DNN(object):
    def __init__(self,model,name,x,y):
        self.path = os.path.join("models",name)
        self.log_path = "logs/{}.pkl".format(name) #用pickle保存训练损失的变化值。
        
        self.name = name
        self.model = model
        self.x = x
        self.y = y
        self.history = None
        
        self.checkpoint_path = os.path.join(self.path,'epoch_{epoch:03d}loss{loss:.3f}.h5')
        #生成模型保存路径,训练记录保存路径
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        if not os.path.exists("logs"):
            os.mkdir("logs")
        
    def fit(self):
        '''
        读取训练的历史数据和模型，或者重新训练模型。
        '''
        if len(os.listdir(self.path))==0:
            #模型文件中模型数为0，则开始训练            
            checkpoint = ModelCheckpoint(self.checkpoint_path,monitor='loss',\
                                 save_best_only=True,save_weights_only=False,verbose=1)
            hist = self.model.fit(x =self.x,y=self.y,batch_size = 32,epochs = 100,callbacks=[checkpoint])
            self.history = hist.history['loss']
            #将训练历史写入logs文件中
            pickle.dump(self.history, open(self.log_path,'wb'))
        else:
            print("load history from file!")
            self.history = pickle.load(open(self.log_path,'rb'))

        #读取最优的训练值的模型。
        model_name = sorted(os.listdir(self.path))[-1]
        print("load {} as best model".format(model_name))
        self.model = keras.models.load_model(os.path.join(self.path,model_name),\
                                custom_objects={'MLE_loss': MLE_loss})
        
    def print_analyse(self):
        #计算标准残差序列，对其做arch检验。
        log_sigma = self.model.predict(self.x)
        sigma = np.sqrt(np.exp(log_sigma))
        sigma = np.reshape(sigma,(-1))
        std_res = self.y/sigma
        
        #检验arch效应
        arch_test = het_arch(std_res)
        print("*"*10,self.name,"*"*10)
        print("{}模型,最优的损失值{:.3f}".format(self.name,min(self.history)))
        print("arch检验的统计量：{:.3f}，对应p值为:{:.4f}\n".format(arch_test[0],arch_test[1]))
        
        
        #绘制训练损失变化图
        plt.plot(np.arange(1,101), self.history, color='black',marker = ',',markersize = 7,\
                     lw=1, label='{} best loss: {:.3f}'.format(self.name,min(self.history)))
        plt.xlabel('interation')     
        plt.ylabel('Training Loss')
        plt.title("{} Training loss with iteration".format(self.name))
        plt.legend(loc="upper right")
        
        #新建一个文件夹，存放所有训练时产生的损失图片。
        if not os.path.exists("plot_images/traning_loss"):
            os.mkdir("plot_images/traning_loss")
        plt.savefig("plot_images/traning_loss/{}.png".format(self.name))
        plt.show()
        plt.close()