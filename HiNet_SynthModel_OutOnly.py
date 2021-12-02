#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:04:44 2019

@author: tao
"""

import os
import torch
import torch.nn as nn
import numpy as np

import time
import datetime

from torch.utils.data import DataLoader

#from models import *
#from fusion_models import *  # revise in 09/03/2019
from dataset import MultiModalityData_load
from funcs.utils import *
import torch.nn as nn
import scipy.io as scio
from torch.autograd import Variable
import torch.autograd as autograd
#import IVD_Net as IVD_Net
import model.syn_model_OutOnly as models
import model.u2net as u2net
import copy
import glob
from torchsummary import summary


#from config import opt
#from visualize import Visualizer
#testing     

#os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'

cuda = True if torch.cuda.is_available() else False
torch.backends.cudnn.benchmark = True
FloatTensor   = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor    = torch.cuda.LongTensor if cuda else torch.LongTensor


class LatentSynthModel():
    
    ########################################################################### 
    
    def __init__(self,opt):
        
        self.opt         = opt  
        self.generator   = models.Multi_modal_generator(1,1,64)
        #self.generator   = u2net.U2NET(1,1)
        self.discrimator = models.Discriminator()
        
        if opt.use_gpu: 
            self.generator    = self.generator.cuda()
            self.discrimator  = self.discrimator.cuda()
                      
        if torch.cuda.device_count() > 1:
            self.generator    = nn.DataParallel(self.generator,device_ids=self.opt.gpu_id)
            self.discrimator  = nn.DataParallel(self.discrimator,device_ids=self.opt.gpu_id)  
        
    ########################################################################### 
    #show the parameters of the network
        #summary(models.Multi_modal_generator(2,1,32),(256,224)) #(model_name,(input shape),batch_size)
    #print(self.generator)

    ########################################################################### 
    def train(self):   
        
        if not os.path.isdir(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/'):
            mkdir_p(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/')
        
        logger = Logger(os.path.join(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/'+'run_log.txt'), title='')
        logger.set_names(['Run epoch','D Loss', 'G Loss'])

        #
        self.generator.apply(weights_init_normal)
        self.discrimator.apply(weights_init_normal)
        print('weights_init_normal')
                
        # Optimizers
        optimizer_D     = torch.optim.Adam(self.discrimator.parameters(), lr=self.opt.lr,betas=(self.opt.b1, self.opt.b2))
        optimizer_G     = torch.optim.Adam(self.generator.parameters(),lr=self.opt.lr,betas=(self.opt.b1, self.opt.b2))

        # Learning rate update schedulers
        lr_scheduler_G  = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.opt.epochs, 0, self.opt.decay_epoch).step)
        lr_scheduler_D  = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(self.opt.epochs, 0, self.opt.decay_epoch).step)
    
            
        # Lossesgenerator
        criterion_GAN   = nn.MSELoss()
        criterion_identity = nn.L1Loss()
        if self.opt.use_gpu:
            criterion_GAN = nn.MSELoss().cuda()
            criterion_identity = nn.L1Loss().cuda()

        # Load data        
        train_data   = MultiModalityData_load(self.opt,train=True)
        train_loader = DataLoader(train_data,batch_size=self.opt.batch_size,shuffle=False)


        batches_done = 0
        prev_time    = time.time()
        # ---------------------------- *training * ---------------------------------
        for epoch in range(self.opt.epochs):      
            for ii, inputs in enumerate(train_loader):
                #print(ii)
                # define diferent synthesis tasks
                [x1,x2,x3] = model_task(inputs,self.opt.task_id) # train different synthesis task
                #print(inputs[0].shape)
                fake = torch.zeros([inputs[0].shape[3] * inputs[0].shape[0], 1, 14, 12], requires_grad=False)
                valid = torch.ones([inputs[0].shape[3] * inputs[0].shape[0], 1, 14, 12], requires_grad=False)
                if self.opt.use_gpu:
                    fake = fake.cuda()
                    valid = valid.cuda()
                              
                ###############################################################                     
                if self.opt.use_gpu:
                    x1   = x1.cuda()
                    x2   = x2.cuda()
                    x3   = x3.cuda()
                    
                x_fu = torch.cat([x1,x2],dim=1)

                # ----------------------
                #  Train generator
                # ----------------------
                optimizer_G.zero_grad()
                
                #d0,d1,d2,d3,d4,d5,d6 = self.generator(x1)
                x_fake = self.generator(x_fu)
                
                # Identity loss
                loss_re = criterion_identity(x_fake, x3)
                '''loss_re0 = criterion_identity(output0, x3)
                loss_re1 = criterion_identity(output1, x3)
                loss_re2 = criterion_identity(output2, x3)
                loss_re3 = criterion_identity(output3, x3)
                loss_re4 = criterion_identity(output4, x3)
                loss_re5 = criterion_identity(output5, x3)'''
                #loss_re6 = criterion_identity(d6, x3)
                #loss_re1 = criterion_identity(x1_re, x3)
                #loss_re2 = criterion_identity(x2_re, x3)
                
  
                # gan loss 
                loss_GAN = criterion_GAN(self.discrimator(x_fake), valid) 
                            
                # total loss
                #loss_G = loss_GAN + 100*loss_re3 + 20*loss_re1 + 20*loss_re2
                loss_G = loss_GAN + 100*loss_re
                #loss_G =  loss_re0 + loss_re1 + loss_re2 + loss_re3 +loss_re4 +loss_re5 + loss_re6
                loss_G.backward(retain_graph=True)

#-------------------------------------------------------------------------------------
                optimizer_D.zero_grad()
                # Real loss
                loss_real = criterion_GAN(self.discrimator(x3), valid)
                #print('loss_real:', loss_real.item())
                loss_fake = criterion_GAN(self.discrimator(x_fake), fake)
                #print('loss_fake:', loss_fake.item())
                # Total loss
                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward(retain_graph=True)

                optimizer_D.step()
                optimizer_G.step()
#-------------------------------------------------------------------------------------
                '''# ----------------------
                #  Train Discriminators
                # ----------------------
                optimizer_D.zero_grad()

                
                # Real loss
                loss_real = criterion_GAN(self.discrimator(x3), valid)
                loss_fake = criterion_GAN(self.discrimator(x_fake), fake)
                # Total loss
                loss_D = (loss_real + loss_fake) / 2

                loss_D.backward(retain_graph=True)
                optimizer_D.step()'''
                
                # time
                batches_left = self.opt.epochs * len(train_loader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / self.opt.n_critic)
                prev_time = time.time()
                    
                    #print('Epoch:', epoch, '| D_loss: %.6f' % loss_D.item(),'| G_loss: %.6f' % loss_G.item())
                if ii%50 == 0:
                
                    print('\r[Epoch %d/%d]:' % (epoch, self.opt.epochs),'[Batch %d/%d]:' % (ii, len(train_loader)), '| D_loss: %.6f' % loss_D.item(), '| G_loss: %.6f' % loss_G.item(),'ETA: %s' %time_left)
            
                batches_done += 1
                
   
            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D.step()

            logger.append([epoch, loss_D.item(), loss_G.item()])
            
            # Save model checkpoints
            if epoch > 0 and (epoch) % self.opt.checkpoint_interval == 0:
                
                torch.save(self.generator.state_dict(),  self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/generator_%d.pkl' % (epoch))
                torch.save(self.discrimator.state_dict(),self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/discrimator_%d.pkl' % (epoch))

    ###########################################################################
    def test(self,ind_epoch):   
         
        self.generator.load_state_dict(torch.load(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/'+ 'generator_'+str(ind_epoch)+'.pkl',map_location=torch.device('cpu')),strict=False)
       
        # Load data        
        te_data   = MultiModalityData_load(self.opt,train=False,test=True)
        te_loader = DataLoader(te_data,batch_size=self.opt.batch_size,shuffle=False)
        
        pred_eva_set = []
        for ii, inputs in enumerate(te_loader): 
            #print(ii) 
            # define diferent synthesis tasks
            T1d_cur_path = str(inputs[3])
            T2d_cur_path = str(inputs[4])
            T1c_cur_path = str(inputs[5])
            [x_in1, x_in2, x_out] = model_task(inputs,self.opt.task_id)
            x_fusion   = torch.cat([x_in1,x_in2],dim=1)
                      
            if self.opt.use_gpu:
                x_fusion     = x_fusion.cuda()
                #x_in1 = x_in1.cuda()
            
            
            # pred_out -- [batch_size*4,1,128,128]
            # x3       -- [batch_size*4,1,128,128]
            pred_out = self.generator(x_fusion) 
            #pred_out0,pred_out1,pred_out2,pred_out3,pred_out4,pred_out5,pred_out6 = self.generator(x_in1)
            #output0,output1,output2,output3,output4,output5 = self.generator(x_fusion)  


            #input_1_images,input_2_images,pred_images0, pred_images1,pred_images2,pred_images3,pred_images4,pred_images5,real_images, errors0,errors1,errors2,errors3,errors4,errors5 = prediction_syn_results(x_in1,x_in2,output0,output1,output2,output3,output4,output5,x_out)
            input_1_images, input_2_images, pred_images,real_images = prediction_syn_results(x_in1,x_in2,pred_out,x_out)
            
            patient = 'p12/'
            npy_type = 'npy_rigid/'  # npy_hospital  npy_rigid
            png_type = 'png_rigid/'  # png_hospital  png_rigid
            
            
            if not os.path.isdir(self.opt.save_path+'/'+patient+npy_type+ 'input_1'):
                mkdir_p(self.opt.save_path+'/'+patient+npy_type + 'input_1')
            np.save(self.opt.save_path+'/'+patient+npy_type+ 'input_1'+'/'+ T1d_cur_path.split('/')[-1].split('.')[0] + '.npy',input_1_images)
            
            input_1_images = np.squeeze((input_1_images-input_1_images.min())/(input_1_images.max()-input_1_images.min()))*255
            input_1_PNG = Image.fromarray(input_1_images)
            if not os.path.isdir(self.opt.save_path+'/'+patient+png_type+ 'input_1'):
                mkdir_p(self.opt.save_path+'/'+patient+png_type+'input_1')
            name = T1d_cur_path.split('/')[-1].split('.')[0]
            input_1_PNG.convert('L').save(self.opt.save_path+'/'+patient+png_type+'input_1/'+name+'.png')
            
            
            if not os.path.isdir(self.opt.save_path+'/'+patient+npy_type+ 'input_2'):
                mkdir_p(self.opt.save_path+'/'+patient+npy_type + 'input_2')
            np.save(self.opt.save_path+'/'+patient+npy_type+'input_2'+'/'+ T2d_cur_path.split('/')[-1].split('.')[0] + '.npy',input_2_images)
            
            input_2_images = np.squeeze((input_2_images-input_2_images.min())/(input_2_images.max()-input_2_images.min()))*255
            input_2_PNG = Image.fromarray(input_2_images)
            if not os.path.isdir(self.opt.save_path+'/'+patient+png_type+ 'input_2'):
                mkdir_p(self.opt.save_path+'/'+patient+png_type+'input_2')
            name = T2d_cur_path.split('/')[-1].split('.')[0]
            input_2_PNG.convert('L').save(self.opt.save_path+'/'+patient+png_type+'input_2/'+name+'.png')
            
            
            if not os.path.isdir(self.opt.save_path+'/'+patient+npy_type+ 'real'):
                mkdir_p(self.opt.save_path+'/'+patient+npy_type + 'real')
            np.save(self.opt.save_path+'/'+patient+npy_type+'real'+'/'+ T1c_cur_path.split('/')[-1].split('.')[0] + '.npy',real_images)
            
            real_PNG = np.squeeze((real_images-real_images.min())/(real_images.max()-real_images.min()))*255
            real_PNG = Image.fromarray(real_PNG)
            if not os.path.isdir(self.opt.save_path+'/'+patient+png_type+ 'real'):
                mkdir_p(self.opt.save_path+'/'+patient+png_type+'real')
            name = T1c_cur_path.split('/')[-1].split('.')[0]
            real_PNG.convert('L').save(self.opt.save_path+'/'+patient+png_type+'real/'+name+'.png')
            
            
            if not os.path.isdir(self.opt.save_path+'/'+patient+npy_type+ 'pred'):
                mkdir_p(self.opt.save_path+'/'+patient+npy_type + 'pred')
            np.save(self.opt.save_path+'/'+patient+npy_type+ 'pred'+'/'+ T1c_cur_path.split('/')[-1].split('.')[0] + '.npy',pred_images)
            
            pred_images = np.squeeze((pred_images-pred_images.min())/(pred_images.max()-pred_images.min()))*255
            pred_PNG = Image.fromarray(pred_images)
            if not os.path.isdir(self.opt.save_path+'/'+patient+png_type+ 'pred'):
                mkdir_p(self.opt.save_path+'/'+patient+png_type+'pred')
            name = T1c_cur_path.split('/')[-1].split('.')[0]
            pred_PNG.convert('L').save(self.opt.save_path+'/'+patient+png_type+'pred/'+name+'.png')

        '''pred_eva_set_temp = []
        pred_eva_set = []
        patient = os.listdir(self.opt.save_path+'/npy/')
        for i in patient:
        #-----each patient-----
            pred = glob.glob(self.opt.save_path+'/npy/'+'/pred/*')
            real = glob.glob(self.opt.save_path+'/npy/'+'/real/*')
            for j in range(len(pred)):
            #-----each slice-----
                assert pred[j].split('/')[-1] == real[j].split('/')[-1], 'slice mis_match, please check carefully!'
                if len(pred[j].split('/')[-1]) < 10:
                  pred_image = np.load(pred[j])
                  real_image = np.load(real[j])
                  errors = ErrorMetrics(pred_image.astype(np.float32),real_image.astype(np.float32))
                  pred_eva_set_temp.append([errors['MAE'],errors['MSE'],errors['SSIM'],errors['PSNR']])
            mean = [np.array(pred_eva_set_temp)[:,0].mean(),np.array(pred_eva_set_temp)[:,1].mean(),np.array(pred_eva_set_temp)[:,2].mean(),np.array(pred_eva_set_temp)[:,3].mean()]
            pred_eva_set_temp = []
            pred_eva_set.append(mean)

        #np.save(self.opt.save_path+'/'+ 'errors_' + str(ind_epoch) + '.npy',pred_eva_set)
           

        mean_values = [ind_epoch,np.array(pred_eva_set)[:,0].mean(),np.array(pred_eva_set)[:,1].mean(),np.array(pred_eva_set)[:,2].mean(),np.array(pred_eva_set)[:,3].mean()]
        std_values = [ind_epoch,np.array(pred_eva_set)[:,0].std(),np.array(pred_eva_set)[:,1].std(),np.array(pred_eva_set)[:,2].std(),np.array(pred_eva_set)[:,3].std()]

        print(mean_values,std_values)'''
    
    
    
