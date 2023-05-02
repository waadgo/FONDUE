# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 13:34:26 2022

@author: walte
"""

import os
import scipy

basename = os.path.join("C:\Denoising","DCCR-Net","variable_maps")

def get_var_map_conf(im):
    shape = im.shape
    conf_shape = shape[0]
    name = os.path.join(basename,str(conf_shape)+'_map')
    try:
        vmap = scipy.io.loadmat(name, variable_names = "map")
    except:
        raise TypeError('Noise map file requested has not been created or could not be opened. Please verify it.')           
    else:
        vmap = vmap['map'][:]
        return vmap

def get_var_map_size(size = 256):
    conf_shape = size
    name = os.path.join(basename,str(conf_shape)+'_map')
    try:
        vmap = scipy.io.loadmat(name, variable_names = "map")
    except:
        raise TypeError('Noise map file requested has not been created or could not be opened. Please verify it.')           
    else:
        vmap = vmap['map'][:]
        return vmap
