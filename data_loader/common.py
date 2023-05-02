# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:59:05 2023

@author: walte
"""

import os
import random
import numpy as np
import imageio
from tqdm import tqdm
import torch
import csv
from collections import OrderedDict


####################
# Resuming training
####################

# def get_best_lr_excel(log, config):
    
    
def csv_to_odict(filename):

    file = open(filename, mode='r')

    csvReader = csv.reader(file)

    # get rid of header row
    header = next(csvReader)
    # print(header)

    odict = OrderedDict()
    for row in csvReader:
        odict[row[0]] = row[1:]
        # print(row)

    return odict


####################
# image processing
# process on numpy image
####################
def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        # if img.shape[2] == 3: # for opencv imread
        #     img = img[:, :, [2, 1, 0]]
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255.)

        return tensor

    return [_np2Tensor(_l) for _l in l]


def get_patch(img_in, img_tar, patch_size, scale):
    ih, iw = img_in.shape[:2]
    oh, ow = img_tar.shape[:2]

    ip = patch_size

    if ih == oh:
        tp = ip
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = ix, iy
    else:
        tp = ip * scale
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar

def center_crop(img_in, img_tar, patch_size=256):
    ch, cw = patch_size, patch_size
    
    img_in_w, img_in_h = img_in.shape[0], img_in.shape[1]
    img_in_c1 = (img_in_w-cw)//2
    img_in_c2 = (img_in_h-ch)//2
    img_in = img_in[img_in_c1:img_in_c1+cw, img_in_c2:img_in_c2+ch,:]
    
    img_tar_w, img_tar_h = img_tar.shape[0], img_tar.shape[1]
    img_tar_c1 = (img_tar_w-cw)//2
    img_tar_c2 = (img_tar_h-ch)//2
    img_tar = img_tar[img_tar_c1:img_tar_c1+cw, img_tar_c2:img_tar_c2+ch,:]

    return img_in, img_tar


def add_noise2(x, noise='.'):
    # if noise is not '.':
    #     condition1 = x.max() <= 1
    #     condition2 = x.min() >= 0
    #     if torch.logical_and(condition1,condition2):
    #         x = x*255.
            
        noise_type = noise[0]
        noise_value = float(noise[1:])/100
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            # noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.single) + noises.astype(np.single)
        x_noise = x_noise.clip(0., 1.).astype(np.single)
        # x_noise = x.astype(np.int16) + noises.astype(np.int16)
        # x_noise = x.astype(np.int16)
        # x_noise = x_noise.astype(np.uint8)
        # x_noise = torch.from_numpy(x_noise)
        
        
        return x_noise
    
def add_noise(x, vmap, noise='.'):
    if noise != '.':
        # condition1 = x.max() <= 1
        # condition2 = x.min() >= 0
        # if torch.logical_and(condition1,condition2):
        #     x = x*255.
            
        noise_type = noise[0]
        noise_value = float(noise[1:])/100
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            x_noise = x.astype(np.single) + noises.astype(np.single)
            x_noise = x_noise.clip(0., 1.).astype(np.single)
        # elif noise_type == 'R':
        #     noiseR = np.random.normal(x * noise_value) / noise_value
        #     noiseR = noiseR - noiseR.mean(axis=0).mean(axis=0)
            
        #     noiseI = np.random.normal(x * noise_value) / noise_value
        #     noiseI = noiseI - noiseI.mean(axis=0).mean(axis=0)
            
        #     x_real = x.cpu().detach().numpy().astype(np.int16) + noiseR.astype(np.int16)
        #     x_real2 = np.absolute(x_real) * np.absolute(x_real)
            
        #     x_imaginary = noiseI.astype(np.int16)
        #     x_imaginary2 = np.absolute(x_imaginary) * np.absolute(x_imaginary)
            
        #     x_noise = np.sqrt(x_real2 + x_imaginary2)
        #     x_noise = x_noise.clip(0., 1.).astype(np.single)
        elif noise_type == 'R':
            vmap = np.moveaxis(vmap, (0, 1, 2), (1, 2, 0))
            vmap = np.expand_dims(vmap, 1)
            noiseR = np.random.normal(scale=noise_value, size=x.shape)*vmap
            noiseI = np.random.normal(scale=noise_value, size=x.shape)*vmap
            
            real_part = (x.astype(np.single) + noiseR.astype(np.single))**2
            imag_part = (noiseI.astype(np.single))**2
            
            x_noise = np.sqrt(real_part + imag_part)
            x_noise = x_noise.clip(0., 1.).astype(np.single)

        return x_noise
    else:
        return x
    


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]

def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def augment_img_np3(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return img.transpose(1, 0, 2)
    elif mode == 2:
        return img[::-1, :, :]
    elif mode == 3:
        img = img[::-1, :, :]
        img = img.transpose(1, 0, 2)
        return img
    elif mode == 4:
        return img[:, ::-1, :]
    elif mode == 5:
        img = img[:, ::-1, :]
        img = img.transpose(1, 0, 2)
        return img
    elif mode == 6:
        img = img[:, ::-1, :]
        img = img[::-1, :, :]
        return img
    elif mode == 7:
        img = img[:, ::-1, :]
        img = img[::-1, :, :]
        img = img.transpose(1, 0, 2)
        return img


def modcrop(img_in, scale):
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [%d].' % img.ndim)
    return img