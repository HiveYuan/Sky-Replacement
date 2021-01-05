import os
import time
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
import torchvision.models as models
from tools.common_tools import set_seed
from torch.utils.tensorboard import SummaryWriter
from tools.my_dataset import SkyDataset
from tools.unet import UNet
from tools.LovaszLoss import lovasz_hinge
from torch.utils.data import SubsetRandomSampler
from color_transfer import color_transfer
import argparse
import cv2
import func
import unet_infer as unet

tic = time.time()

print('launching...')

img_num='999'
sky_num='777'
show=1
save_result=0

img_path='d:/MyLearning/DIP/Final_Project/sky_replacement/Unet/Demo/'+img_num+'.png'
# mask_path='d:/MyLearning/DIP/Final_Project/Unet/results/0_pred_gray_img.png'
sky_path='d:/MyLearning/DIP/Final_Project/sky_replacement/Unet/sky/'+sky_num+'.jpg'

original_img=cv2.imread(img_path)
sky=cv2.imread(sky_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
sky = cv2.cvtColor(sky, cv2.COLOR_BGR2RGB)

img,mask_bw=unet.unet_infer(img_path,1,0)
sz=original_img.shape
print('replacing sky...')
I_rep=func.replace_sky(img,mask_bw,sky)

print('color transferring...')
transfer = func.color_transfer(sky,mask_bw,I_rep,1)
mask_edge=cv2.Canny(mask_bw,100,200)
mask_edge_hwc=cv2.merge([mask_edge, mask_edge, mask_edge])
print('guided filtering...')
final =func.guideFilter(img,transfer,mask_edge_hwc,(8,8),0.01)

toc = time.time()
print('Done! Total time is '+str(toc-tic)+'s')

if show:
    plt.subplot(231).imshow(original_img)
    plt.title('Original Image')
    plt.subplot(232).imshow(mask_bw,cmap="gray")
    plt.title('Mask of Sky')
    plt.subplot(233).imshow(sky)
    plt.title('Chosen Sky')
    plt.subplot(234).imshow(I_rep)
    plt.title('Sky-Replaced Image')
    plt.subplot(235).imshow(transfer)
    plt.title('Color Transferred')
    plt.subplot(236).imshow(final)
    plt.title('Guide Filtered')
    plt.show()

if save_result:
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    mask_bw = cv2.cvtColor(mask_bw, cv2.COLOR_RGB2BGR)
    sky = cv2.cvtColor(sky, cv2.COLOR_RGB2BGR)
    I_rep = cv2.cvtColor(I_rep, cv2.COLOR_RGB2BGR)
    transfer = cv2.cvtColor(transfer, cv2.COLOR_RGB2BGR)
    final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
    cv2.imwrite('d:/MyLearning/DIP/Final_Project/Unet/Report/show/1.png',original_img)
    cv2.imwrite('d:/MyLearning/DIP/Final_Project/Unet/Report/show/2.png',mask_bw)
    cv2.imwrite('d:/MyLearning/DIP/Final_Project/Unet/Report/show/3.png',sky)
    cv2.imwrite('d:/MyLearning/DIP/Final_Project/Unet/Report/show/4.png',I_rep)
    cv2.imwrite('d:/MyLearning/DIP/Final_Project/Unet/Report/show/5.png',transfer)
    cv2.imwrite('d:/MyLearning/DIP/Final_Project/Unet/Report/show/6.png',final)
