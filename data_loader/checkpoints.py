# -*- coding: utf-8 -*-
"""
Created on Sat May  7 16:02:01 2022
https://gist.github.com/rachit221195/492768a992fa2f69c0d9769f18291855#file-save_ckp-py
https://gist.github.com/rachit221195/91d5b6e96f5d268af8842235529f88c2#file-load_ckp-py
@author: walte
"""

import torch
import shutil
from glob import glob

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    ckpt_name, best_model_name = get_ckp_names(state['epoch'])
    f_path = checkpoint_dir + ckpt_name
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + best_model_name
        shutil.copyfile(f_path, best_fpath)
        
def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_loss = checkpoint['best_loss']
    return model, optimizer, scheduler, checkpoint['epoch'], best_loss

def load_model(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def get_ckp_names(epoch):
    if epoch >=0 and epoch<10:
        ckpt_name = '/checkpoint_00'+str(epoch)+'.pt'
        best_model_name = '/best_model_00'+str(epoch)+'.pt'
    elif epoch>=10 and epoch<100:
        ckpt_name = '/checkpoint_0'+str(epoch)+'.pt'
        best_model_name = '/best_model_0'+str(epoch)+'.pt'
    else:
        ckpt_name = '/checkpoint_'+str(epoch)+'.pt'
        best_model_name = '/best_model_'+str(epoch)+'.pt'
    return ckpt_name, best_model_name

def get_last_ckp_path(config):
    path = config['ckp_path']
    models_filepath =  path + '/*.pt'    
    model_list = glob(models_filepath)
    num_models = len(model_list)
    last_ckp_path=model_list[num_models-1]
    return last_ckp_path

# def get_best_ckp_path(config):
#     path = config.model_path
#     name = config.name
#     models_filepath =  path + '/'+name+'.pt'    
#     model_list = glob(models_filepath)
#     num_models = len(model_list)
#     last_ckp_path=model_list[num_models-1]
#     return last_ckp_path
def get_best_ckp_path(config):
    path = config.model_path
    # name = config.name
    # models_filepath =  path + '/'+name+'.pt'    
    return path


def get_best_ckp_path_first_stage(config):
    path_first = config.model_path_first_stage
    # path_second = config.model_path_second_stage
    # name = config.name
    # models_filepath =  path + '/'+name+'.pt'    
    return path_first

def get_best_ckp_path_second_stage(config, second_stage_type):
    if second_stage_type == "low":
        path = config.model_path_second_stage_low
    elif second_stage_type == "mid":
        path = config.model_path_second_stage_mid
    # name = config.name
    # models_filepath =  path + '/'+name+'.pt'    
    return path