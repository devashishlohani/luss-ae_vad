import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
import random
from math import log10
#import kornia

def loss_func_rgan(recon_x, x):
    msssim = ((1-pytorch_msssim.ms_ssim(x,recon_x)))/2
    f1 =  F.l1_loss(recon_x, x)
    # psnr_error=(10 * log10( 65025/ ((torch.abs(torch.sum(x) - torch.sum(recon_batch))))))
    psnr_error=(10 * log10( 65025/ ((torch.abs(torch.sum(x) - torch.sum(recon_x))))))

    return msssim + f1 + psnr_error


def psnr_mod(input_volume):
    ### Input volume shape: (1, 16 ,256 ,256), i.e. 4 dims
    max2_per_frame = torch.max(torch.max(input_volume, 3).values, 2).values
    mse_per_frame = torch.mean(torch.mean(input_volume, 3), 2)
    psnr_per_frame = 10 * torch.log10(max2_per_frame / mse_per_frame)
    psnr_vid = torch.mean(psnr_per_frame)
    return psnr_vid

def get_memory_loss(memory_att):
    """The memory attribute should be with size [batch_size, memory_dim, reduced_time_dim, f_h, f_w]
    loss = \sum_{t=1}^{reduced_time_dim} (-mem) * (mem + 1e-12).log() 
    averaged on each pixel and each batch
    2. average over batch_size * fh * fw
    """
    s = memory_att.shape # (2,2000,2,16,16)
    memory_att = (-memory_att) * (memory_att + 1e-12).log()  # [batch_size, memory_dim, time, fh, fw]
    memory_att = memory_att.sum() / (s[0] * s[-2] * s[-1]) 
    return memory_att 

def score_sum_3_lists(list1, list2, list3, alpha, beta, gamma):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha * list1[i] + beta * list2[i] + gamma * list3[i]))

    return list_result

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

def filter(data, template, radius=5):
    arr=np.array(data)
    length=arr.shape[0]  
    newData=np.zeros(length) 

    for j in range(radius//2,arr.shape[0]-radius//2):
        t=arr[ j-radius//2:j+radius//2+1]
        a=np.multiply(t,template)
        newData[j]=a.sum()
    # expand
    for i in range(radius//2):
        newData[i]=newData[radius//2]
    for i in range(-radius//2,0):
        newData[i]=newData[-radius//2]    
    # import pdb;pdb.set_trace()
    return newData

def calc(r=5, sigma=2):
    k = np.zeros(r)
    for i in range(r):
        k[i] = 1/((2*math.pi)**0.5*sigma)*math.exp(-((i-r//2)**2/2/(sigma**2)))
    return k

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):

    return 10 * math.log10(1 / mse)

def psnrv2(mse, peak):
    # note: peak = max(I) where I ranged from 0 to 2 (considering mse is calculated when I is ranged -1 to 1)
    return 10 * math.log10(peak * peak / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)  # +1/2 probably the g() function. Normalize from 0-1. Although not exactly min() and max() value.
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score
    
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_v2(psnr_list):
    from sklearn.preprocessing import MinMaxScaler
    data = np.array(psnr_list).reshape(-1, 1)
    rs = MinMaxScaler().fit(data)
    anomaly_score_list = list(rs.transform(data).reshape(-1))

    return anomaly_score_list


def anomaly_scorev2(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_inv2(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result

def pse_clip(r_clip, gauss):
    x_blur: torch.tensor = gauss(r_clip.float())
    pse = torch.mean(torch.square(x_blur),(1,2,3)).tolist()
    return pse

def mse_clip(r_clip):
    mse = torch.mean(torch.square(r_clip),(1,2,3)).tolist()
    return mse

def multi_scale_ge_clip(r_clip, s_factors=[0.7]):
    msge = np.zeros(r_clip.shape[0])
    clip_data = r_clip.float()
    for f in s_factors:
        x_blur = kornia.geometry.transform.rescale(clip_data, factor=f, interpolation='bilinear', antialias=True)
        ge = torch.mean(torch.square(x_blur),(1,2,3)).cpu().detach().numpy()
        msge += ge
    return list(msge)
