#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:33:55 2019

@author: tao
"""
import os
import scipy.io as sio 
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from numpy.lib.stride_tricks import as_strided as ast
#from skimage.measure import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error as mae
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image


def prediction_syn_results(input_1,input_2,pred_out,real_out):
#def prediction_syn_results(input_1,input_2,pred_out,real_out):
    
    ###########################################################################
    ### Note that
    # there is two manners to evaluate the testing sets
    # for example, using T1 + T2 to synthesize Flair
    # -->(1) the ground truths of Flair keep original size ([160,180,batch_size]) without spliting into small pathces (128*128). In this case, the 
    # synthesized results with size [batch_size*num_patch,1,128,128]， we need change it to [160,180,batch_size]
     
    # -->(2) the ground truths and synthesized results are all with size [batch_size*num_patch,1,128,128]， we need change 
    # them to [160,180,batch_size]. See details of this maner below.
    
    # When one volume as input, we set batch_size=num_slice
        
    ###########################################################################    

    
    # [batch_size*num_patch,1,128,128] -- > [batch_size, num_patch, 128, 128]
    input_1_res = torch.reshape(input_1,(1,256,224))
    input_2_res = torch.reshape(input_2,(1,256,224))
    real_res = torch.reshape(real_out,(1,256,224))
    pred_res = torch.reshape(pred_out,(1,256,224))
    '''pred_res0 = torch.reshape(pred_out0,(1,256,224))
    pred_res1 = torch.reshape(pred_out1,(1,256,224))
    pred_res2 = torch.reshape(pred_out2,(1,256,224))
    pred_res3 = torch.reshape(pred_out3,(1,256,224))
    pred_res4 = torch.reshape(pred_out4,(1,256,224))
    pred_res5 = torch.reshape(pred_out5,(1,256,224))'''
    #pred_res6 = torch.reshape(pred_out6,(1,256,224))
        
    input_1_images = input_1_res.cpu().detach().numpy()
    input_2_images = input_2_res.cpu().detach().numpy()
    real_images = real_res.detach().numpy() 
    pred_images = pred_res.cpu().detach().numpy()
    '''pred_images0 = pred_res0.cpu().detach().numpy()#pred_res.cpu().detach().numpy()
    pred_images1 = pred_res1.cpu().detach().numpy()
    pred_images2 = pred_res2.cpu().detach().numpy()
    pred_images3 = pred_res3.cpu().detach().numpy()
    pred_images4 = pred_res4.cpu().detach().numpy()
    pred_images5 = pred_res5.cpu().detach().numpy()'''
    #pred_images6 = pred_res6.cpu().detach().numpy()
    #print(pred_images.max(),pred_images.min())
    #print(pred_images.shape)
    #print(real_images.shape)
    input_1_images = (input_1_images+1)/2*3071 #4686
    input_2_images = (input_2_images+1)/2*1762 #3411
    real_images = (real_images+1)/2*4095
    pred_images = (pred_images+1)/2*4095
    '''pred_images0 = (pred_images0+1)/2*4095
    pred_images1 = (pred_images1+1)/2*4095
    pred_images2 = (pred_images2+1)/2*4095
    pred_images3 = (pred_images3+1)/2*4095
    pred_images4 = (pred_images4+1)/2*4095
    pred_images5 = (pred_images5+1)/2*4095'''
    #pred_images6 = (pred_images6+1)/2*4095
    
    
    errors = ErrorMetrics(pred_images.astype(np.float32), real_images.astype(np.float32)) 
    '''errors0 = ErrorMetrics(pred_images0.astype(np.float32), real_images.astype(np.float32)) 
    errors1 = ErrorMetrics(pred_images1.astype(np.float32), real_images.astype(np.float32)) 
    errors2 = ErrorMetrics(pred_images2.astype(np.float32), real_images.astype(np.float32)) 
    errors3 = ErrorMetrics(pred_images3.astype(np.float32), real_images.astype(np.float32)) 
    errors4 = ErrorMetrics(pred_images4.astype(np.float32), real_images.astype(np.float32)) 
    errors5 = ErrorMetrics(pred_images5.astype(np.float32), real_images.astype(np.float32)) '''
    #errors6 = ErrorMetrics(pred_images6.astype(np.float32), real_images.astype(np.float32)) 
        
    #return input_1_images,input_2_images, pred_images0, pred_images1,pred_images2,pred_images3,pred_images4,pred_images5,real_images, errors0,errors1,errors2,errors3,errors4,errors5
    return input_1_images, input_2_images, pred_images,real_images
 


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    

class Logger(object):
	'''Save training process to log file with simple plot function.'''
	def __init__(self, fpath, title=None, resume=False): 
		self.file = None
		self.resume = resume
		self.title = '' if title == None else title
		if fpath is not None:
			if resume: 
				self.file = open(fpath, 'r') 
				name = self.file.readline()
				self.names = name.rstrip().split('\t')
				self.numbers = {}
				for _, name in enumerate(self.names):
					self.numbers[name] = []

				for numbers in self.file:
					numbers = numbers.rstrip().split('\t')
					for i in range(0, len(numbers)):
						self.numbers[self.names[i]].append(numbers[i])
				self.file.close()
				self.file = open(fpath, 'a')  
			else:
				self.file = open(fpath, 'w')

	def set_names(self, names):
		if self.resume: 
			pass
		# initialize numbers as empty list
		self.numbers = {}
		self.names = names
		for _, name in enumerate(self.names):
			self.file.write(name)
			self.file.write('\t')
			self.numbers[name] = []
		self.file.write('\n')
		self.file.flush()


	def append(self, numbers):
		assert len(self.names) == len(numbers), 'Numbers do not match names'
		for index, num in enumerate(numbers):
			self.file.write("{0:.6f}".format(num))
			self.file.write('\t')
			self.numbers[self.names[index]].append(num)
		self.file.write('\n')
		self.file.flush()

	def plot(self, names=None):   
		names = self.names if names == None else names
		numbers = self.numbers
		for _, name in enumerate(names):
			x = np.arange(len(numbers[name]))
			plt.plot(x, np.asarray(numbers[name]))
		plt.legend([self.title + '(' + name + ')' for name in names])
		plt.grid(True)

	def close(self):
		if self.file is not None:
			self.file.close()
             

class AverageMeter(object):
	"""Computes and stores the average and current value
	   Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
	"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		
	def avg(self):
		return self.sum / self.count

def mkdir_p(path):
	'''make dir if not exist'''
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise    
            

        
def model_task(inputs,task_id):
    ####################   0: t1  1: t2  2:t1c
    if task_id == 1:
        in_id1 = 0
        in_id2 = 1
        out_id = 2
    #print(np.array(inputs[0]).shape)
    x1 = torch.reshape(inputs[in_id1],[1,1,256,224]).type(torch.FloatTensor) 
    x2 = torch.reshape(inputs[in_id2],[1,1,256,224]).type(torch.FloatTensor) 
    x3 = torch.reshape(inputs[out_id],[1,1,256,224]).type(torch.FloatTensor) 
    
    #x1 = torch.reshape(inputs[in_id1], [inputs[in_id1].shape[3]*inputs[in_id1].shape[0],1,inputs[in_id1].shape[2],inputs[in_id1].shape[1]]).type(torch.FloatTensor)        
    #x2 = torch.reshape(inputs[in_id2], [inputs[in_id2].shape[3]*inputs[in_id2].shape[0],1,inputs[in_id2].shape[2],inputs[in_id2].shape[1]]).type(torch.FloatTensor)
    #x3 = torch.reshape(inputs[out_id], [inputs[out_id].shape[3]*inputs[out_id].shape[0],1,inputs[out_id].shape[2],inputs[out_id].shape[1]]).type(torch.FloatTensor)
    #print(np.array(x1).shape)
    #np.save('/home/wen/桌面/x1.npy',np.array(x1))
    return x1,x2,x3

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
        
def ErrorMetrics(vol_s, vol_t):
    
    # calculate various error metrics.
    # vol_s should be the synthesized volume (a 3d numpy array) or an array of these volumes
    # vol_t should be the ground truth volume (a 3d numpy array) or an array of these volumes

#    vol_s = np.squeeze(vol_s)
#    vol_t = np.squeeze(vol_t)
    
#    vol_s = vol_s.numpy()
#    vol_t = vol_t.numpy()

    assert len(vol_s.shape) == len(vol_t.shape) == 3
    assert vol_s.shape[0] == vol_t.shape[0]
    assert vol_s.shape[1] == vol_t.shape[1]
    assert vol_s.shape[2] == vol_t.shape[2]

    #vol_s[vol_t == 0] = 0    ###############################################################################
    #vol_s[vol_s < 0] = 0

    errors = {}
    
    vol_s = vol_s.astype(np.float32)
      
    # errors['MSE'] = np.mean((vol_s - vol_t) ** 2.)
    #errors['MSE'] = np.sum((vol_s - vol_t) ** 2.) / np.sum(vol_t**2)#########   ????????????????????????????????????????
    errors['MAE'] = mae(np.squeeze(vol_s), np.squeeze(vol_t))
    errors['MSE'] = mse(vol_s, vol_t)
    errors['SSIM'] = ssim(np.squeeze(vol_s), np.squeeze(vol_t), data_range=4095)
    #dr = np.max([vol_s.max(), vol_t.max()]) - np.min([vol_s.min(), vol_t.min()])
    errors['PSNR'] = psnr(vol_t, vol_s, data_range=4095)         

#    # non background in both
#    non_bg = (vol_t != vol_t[0, 0, 0])
#    errors['SSIM_NBG'] = ssim(vol_t[non_bg], vol_s[non_bg])
#    dr = np.max([vol_t[non_bg].max(), vol_s[non_bg].max()]) - np.min([vol_t[non_bg].min(), vol_s[non_bg].min()])
#    errors['PSNR_NBG'] = psnr(vol_t[non_bg], vol_s[non_bg], dynamic_range=dr)
#
#    vol_s_non_bg = vol_s[non_bg].flatten()
#    vol_t_non_bg = vol_t[non_bg].flatten()
#    
#    # errors['MSE_NBG'] = np.mean((vol_s_non_bg - vol_t_non_bg) ** 2.)
#    errors['MSE_NBG'] = np.sum((vol_s_non_bg - vol_t_non_bg) ** 2.) /np.sum(vol_t_non_bg**2)

    return errors
