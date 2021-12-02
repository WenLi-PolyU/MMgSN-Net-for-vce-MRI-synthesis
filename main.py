#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:15:11 2019

@author: tao
"""

from HiNet_SynthModel_OutOnly import LatentSynthModel
from config import opt
import fire

    
def train(**kwargs):
    
    opt.parse(kwargs)
    
    SynModel = LatentSynthModel(opt=opt)
    SynModel.train() 
    

def test(**kwargs):
    
    opt.parse(kwargs)
    SynModel = LatentSynthModel(opt=opt)
    #SynModel.test(epochs) 
    SynModel.test(198) 

        
   
if __name__ == '__main__':
    
    #fire.Fire()
    #train()
    test()
    #mean = []
    #std  = []
    #for epochs in range(0,400,2):
    #  test()
