#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus April 18 17:18:50 2019

@author: tao
"""
#coding:utf8
import os
from torch.utils import data
import numpy as np
from torchvision import  transforms as T
from funcs.utils import *
import torch
import scipy.io as scio

  
def loadSubjectData(path):
    
    data = np.load(path).astype(np.float32)
    #data = torch.from_numpy(data.astype(np.float32))
    
    return data


class MultiModalityData_load(data.Dataset):
    
    def __init__(self,opt,transforms=None,train=True,test=False):
        
        self.opt   = opt
        self.test  = test
        self.train = train
        
        if self.test:
            T1c_path_test  = opt.data_path + 'without_mask/p12/T1c/' # p15
            T1c_paths  = [os.path.join(T1c_path_test,i) for i in os.listdir(T1c_path_test)]
            T1d_path_test  = opt.data_path + 'without_mask/p12/T1_rigid/' # T1_rigid  T1
            T1d_paths  = [os.path.join(T1d_path_test,i) for i in os.listdir(T1d_path_test)]
            T2d_path_test  = opt.data_path + 'without_mask/p12/T2_rigid/'  # T2_rigid  T2
            T2d_paths  = [os.path.join(T2d_path_test,i) for i in os.listdir(T2d_path_test)]
            
        if self.train:
            T1c_path_train  = opt.data_path + 'without_mask/train_npy_all/T1c/'
            T1c_paths  = [os.path.join(T1c_path_train,i) for i in os.listdir(T1c_path_train)]
            T1d_path_train  = opt.data_path + 'without_mask/train_npy_all/T1d/'
            T1d_paths  = [os.path.join(T1d_path_train,i) for i in os.listdir(T1d_path_train)]
            T2d_path_train  = opt.data_path + 'without_mask/train_npy_all/T2d/'
            T2d_paths  = [os.path.join(T2d_path_train,i) for i in os.listdir(T2d_path_train)]
   
        #T1c_data_paths      = sorted(T1c_paths,key=lambda x:int(x.split('_')[-2]))
        #T1d_data_paths      = sorted(T1d_paths,key=lambda x:int(x.split('_')[-2]))
        #T2d_data_paths      = sorted(T2d_paths,key=lambda x:int(x.split('_')[-2]))
        T1c_data_paths      = sorted(T1c_paths)
        T1d_data_paths      = sorted(T1d_paths)
        T2d_data_paths      = sorted(T2d_paths)
        self.T1c_data_paths = np.array(T1c_data_paths)
        self.T1d_data_paths = np.array(T1d_data_paths)
        self.T2d_data_paths = np.array(T2d_data_paths)
                
        
    def __getitem__(self,index):
        
        # path
        T1c_cur_path  = self.T1c_data_paths[index]
        T1d_cur_path  = self.T1d_data_paths[index]
        T2d_cur_path  = self.T2d_data_paths[index]
        #print(T1c_cur_path)
        #print(T1d_cur_path)
        #print(T2d_cur_path)
        
        # get images
        data_T1c = loadSubjectData(T1c_cur_path)
        data_T1d = loadSubjectData(T1d_cur_path)
        data_T2d = loadSubjectData(T2d_cur_path)
        data_T1c = data_T1c.reshape([1,256,224])
        data_T1d = data_T1d.reshape([1,256,224])
        data_T2d = data_T2d.reshape([1,256,224])
        
        #normalize to   [-1,1], then normalize
        data_T1c = data_T1c/4095*2-1
        #data_T1c = (data_T1c-(-0.8985570528440848))/0.11645533129867561
        
        #data_T1d= data_T1d/4686*2-1 #original
        data_T1d= data_T1d/4686*2-1
        #data_T1d = (data_T1d-(-0.9047217568082017))/0.12486843032955916
        
        #data_T2d= data_T2d/3411*2-1 #original
        data_T2d= data_T2d/3411*2-1
        #data_T2d = (data_T2d-(-0.9371561442119687))/0.09262640071640432
        
        

        return data_T1d, data_T2d, data_T1c, T1d_cur_path, T2d_cur_path, T1c_cur_path
    
    
    
    def __len__(self):
        return len(self.T1c_data_paths)
    
     
