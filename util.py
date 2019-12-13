# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 08:50:48 2019

@author: a
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_four_subgraph(model_names,hist_loss,colors,shape):
    #绘制在一张图中
    for model_name,loss,color,s in zip(model_names,hist_loss,colors,shape):
        plt.plot(np.arange(1,101), loss, color=color,marker = s,markersize = 7,\
                     lw=1, label='{} best loss: {:.3f}'.format(model_name,min(loss)))
    plt.xlabel('interation')
    plt.ylabel('Training Loss')
    plt.title("Training loss")
    plt.legend(loc="upper right")
    plt.savefig("plot_images/one_training_loss.png")
    plt.show()
    plt.close()
    
    #绘制在四张小图中
    plt.figure(figsize = (12,8))
    for i in range(4):
        plt.subplot(221+i)
        plt.plot(np.arange(1,101), hist_loss[i], color=colors[i],marker = shape[i],markersize = 7,\
                     lw=1, label='{} best loss: {:.3f}'.format(model_names[i],min(hist_loss[i])))
        plt.xlabel('interation')     
        plt.ylabel('Training Loss')
        plt.title("{} Training loss with iteration".format(model_names[i]))
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("plot_images/four_subgraph_training_loss.png")
    plt.show()
    plt.close()

