'''
===============================================================================
Custom Augmentations:
- SinglePlantRotation
- PutPlantOnNewBG
===============================================================================
'''

import torch
import torchvision
import numpy as np
from PIL import Image
# import os
from utils import utils
import matplotlib.pyplot as plt
import random


class SinglePlantRotation(torch.nn.Module):
    def __init__(self, bg_dir, spec_aug_fg, spec_aug_bg, norm_mean, norm_std, img_size):
        super().__init__()
        self.bg_path = utils.getListOfImgFiles(bg_dir)
        self.spec_aug_fg = spec_aug_fg
        self.spec_aug_bg = spec_aug_bg
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.img_size = img_size
        
        # # background augmentations
        bg_transforms = [
            torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transormation
            torchvision.transforms.Normalize(norm_mean, norm_std),
            torchvision.transforms.Resize(size=(img_size,img_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ]
        if self.spec_aug_bg:
            bg_transforms.append(torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5))
        self.bg_transforms = torchvision.transforms.Compose(bg_transforms)
       
        # # foreground (-> plant) augmentations
        fg_transforms = [
            torchvision.transforms.RandomRotation(360)
            ]
        self.fg_transforms = torchvision.transforms.Compose(fg_transforms)
        
        spec_aug_fg_transforms = []
        if self.spec_aug_fg:
            spec_aug_fg_transforms.append(torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5))
        self.spec_aug_fg_transforms = torchvision.transforms.Compose(spec_aug_fg_transforms)
        
    def get_plant_mask(self, im, th):
            rgbvi = ((im[1,:,:]*im[1,:,:])-(im[0,:,:]*im[2,:,:]))/((im[1,:,:]*im[1,:,:])+(im[0,:,:]*im[2,:,:]))
            rgbvi[rgbvi<th]=0
            rgbvi[rgbvi>=th]=1
            return rgbvi
    
    def put_mask_on_bg(self,bg,im,mask):
        bg[:,mask==1]=im[:,mask==1]
        return bg
            
    def __call__(self, im):
        # deNorm = utils.DeNormalize(self.norm_mean, self.norm_std)
        # trans = torchvision.transforms.ToPILImage()
        
        # # rotate image randomly
        im_rot = self.fg_transforms(im)
        # fig, axs = plt.subplots(1, 1)
        # axs.imshow(trans(deNorm(im_rot)))
        # plt.show()
            
        # # get mask from rotated image
        mask_rot = self.get_plant_mask(im_rot, 0.25)
        # fig, axs = plt.subplots(1, 1)
        # axs.imshow(trans(deNorm(mask_rot)))
        # plt.show()
        
        # apply other augmentation on im_rot after calculating the mask
        im_rot = self.spec_aug_fg_transforms(im_rot)
        # fig, axs = plt.subplots(1, 1)
        # axs.imshow(trans(deNorm(im_rot)))
        # plt.show()
        
        # # load and transform random bg image
        bg = self.bg_transforms(Image.open(self.bg_path[random.randint(0,len(self.bg_path)-1)]))
        # fig, axs = plt.subplots(1, 1)
        # axs.imshow(trans(deNorm(bg)))
        # plt.show()
        
        # # overlay bg with rotated img where mask indicates a plant pixel
        new = self.put_mask_on_bg(bg,im_rot,mask_rot)
        # fig, axs = plt.subplots(1, 1)
        # axs.imshow(trans(deNorm(new)))
        # plt.show()
        
        return new
    

class PutPlantOnNewBG(torch.nn.Module):
    '''
    same as SinglePlantRotation(), but without rotation ;)
    '''    
    def __init__(self, bg_dir, spec_aug_fg, spec_aug_bg, norm_mean, norm_std, img_size):
        super().__init__()
        self.bg_path = utils.getListOfImgFiles(bg_dir)
        self.spec_aug_fg = spec_aug_fg
        self.spec_aug_bg = spec_aug_bg
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.img_size = img_size
        
        # # background augmentations
        bg_transforms = [
            torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transormation
            torchvision.transforms.Normalize(norm_mean, norm_std),
            torchvision.transforms.Resize(size=(img_size,img_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ]
        if self.spec_aug_bg:
            bg_transforms.append(torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5))
        self.bg_transforms = torchvision.transforms.Compose(bg_transforms)
        
        spec_aug_fg_transforms = []
        if self.spec_aug_fg:
            spec_aug_fg_transforms.append(torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5))
        self.spec_aug_fg_transforms = torchvision.transforms.Compose(spec_aug_fg_transforms)
        
    def get_plant_mask(self, im, th):
            rgbvi = ((im[1,:,:]*im[1,:,:])-(im[0,:,:]*im[2,:,:]))/((im[1,:,:]*im[1,:,:])+(im[0,:,:]*im[2,:,:]))
            rgbvi[rgbvi<th]=0
            rgbvi[rgbvi>=th]=1
            return rgbvi
    
    def put_mask_on_bg(self,bg,im,mask):
        bg[:,mask==1]=im[:,mask==1]
        return bg
            
    def __call__(self, im):
        # deNorm = utils.DeNormalize(self.norm_mean, self.norm_std)
        # trans = torchvision.transforms.ToPILImage()
                
        # # get mask from image
        mask = self.get_plant_mask(im, 0.25)
        # fig, axs = plt.subplots(1, 1)
        # axs.imshow(trans(deNorm(mask_rot)))
        # plt.show()
        
        # apply other augmentation on im_rot after calculating the mask
        im_rot = self.spec_aug_fg_transforms(im)
        # fig, axs = plt.subplots(1, 1)
        # axs.imshow(trans(deNorm(im_rot)))
        # plt.show()
        
        # # load and transform random bg image
        bg = self.bg_transforms(Image.open(self.bg_path[random.randint(0,len(self.bg_path)-1)]))
        # fig, axs = plt.subplots(1, 1)
        # axs.imshow(trans(deNorm(bg)))
        # plt.show()
        
        # # overlay bg with rotated img where mask indicates a plant pixel
        new = self.put_mask_on_bg(bg,im_rot,mask)
        # fig, axs = plt.subplots(1, 1)
        # axs.imshow(trans(deNorm(new)))
        # plt.show()
        
        return new
        