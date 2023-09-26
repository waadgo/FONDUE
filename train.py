import argparse
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from data_loader.augmentation import AugmentationPadImage
from data_loader.get_varying_maps import get_var_map_size
from data_loader.load_neuroimaging_data_final import AsegDatasetWithAugmentation, MyBatchSampler, get_thick_slices_maponly
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from imageio import imwrite as save
from data_loader.checkpoints import save_ckp, load_ckp, get_last_ckp_path
from shutil import copytree as copyf
import archs
import losses
from utils import AverageMeter, str2bool
import numpy as np
from torch.utils.tensorboard import SummaryWriter

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__


def parse_args():
    print('sys.argv[0] =', sys.argv[0])             
    pathname = os.path.dirname(sys.argv[0])        
    print('path =', pathname)
    print('full path =', os.path.abspath(pathname))
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', 
                        default = os.path.join(r'C:/Denoising/DCCR-Net_datasets/DCCR-Net_v15_1/'),
                        help="Path to the folder containing the hdf5 files of the datasets.")
    parser.add_argument('--name', default='DCCR-Net_v15_1_fine_low_high_noise',
                        help='It will further train DCCR-Net another 6 epochs using very low noise and very low LR. This uses best_checkpoint_001.pt as starting point from the continue_training VERSION, which is after 2 epochs of continue_training (starting epoch would be 8th epoch')
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DCCRNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: DCCRNet)')
    parser.add_argument('--deep_supervision', default=True, type=str2bool)
    parser.add_argument('--input_channels', default=7, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    # loss
    parser.add_argument('--loss', default='VGGWPL',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: VGGWPL)')
    
    # dataset
    parser.add_argument('--dataset', default='DCCR-Net_v15_1',
                        help='dataset name')
    
    auxparser = parser.parse_args()
    dataset_path = auxparser.dataset_path
    parser.add_argument('--hdf5_name_train1', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_trainset1.hdf5"),
                        help='path to training hdf5-dataset')
    parser.add_argument('--hdf5_name_train2', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_trainset2.hdf5"),
                        help='path to training hdf5-dataset')
    parser.add_argument('--hdf5_name_train3', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_trainset3.hdf5"),
                        help='path to training hdf5-dataset')
    parser.add_argument('--hdf5_name_train4', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_trainset4.hdf5"),
                        help='path to training hdf5-dataset')
    parser.add_argument('--hdf5_name_train5', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_trainset5.hdf5"),
                        help='path to training hdf5-dataset')
    parser.add_argument('--hdf5_name_train6', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_trainset6.hdf5"),
                        help='path to training hdf5-dataset')
    parser.add_argument('--hdf5_name_train7', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_trainset7.hdf5"),
                        help='path to training hdf5-dataset')
    parser.add_argument('--hdf5_name_val1', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_testset1.hdf5"),
                        help='path to validation hdf5-dataset')
    parser.add_argument('--hdf5_name_val2', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_testset2.hdf5"),
                        help='path to validation hdf5-dataset')
    parser.add_argument('--hdf5_name_val3', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_testset3.hdf5"),
                        help='path to validation hdf5-dataset')
    parser.add_argument('--hdf5_name_val4', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_testset4.hdf5"),
                        help='path to validation hdf5-dataset')
    parser.add_argument('--hdf5_name_val5', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_testset5.hdf5"),
                        help='path to validation hdf5-dataset')
    parser.add_argument('--hdf5_name_val6', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_testset6.hdf5"),
                        help='path to validation hdf5-dataset')
    parser.add_argument('--hdf5_name_val7', type=str, default=os.path.join(dataset_path,"DCCR-Net_v15_1_testset7.hdf5"),
                        help='path to validation hdf5-dataset')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--validation_batch_size', type=int, default=2, metavar='N',
                        help='input batch size for validation (default: 16)')
        
    #=========== IMAGES OPTIONS ==========    
    parser.add_argument('--save_imgs_path', type=str, default=os.path.join(pathname,"training_imgs"),
                        help='path to validation hdf5-dataset')
    parser.add_argument('--save_imgs_path_val', type=str, default=os.path.join(pathname,"validation_imgs"),
                        help='path to validation hdf5-dataset')
    parser.add_argument('--verbose_save_imgs_train', type=int, default=100, metavar='N',
                        help='every how many mini-batches images will be saved in training')
    parser.add_argument('--verbose_save_imgs_val', type=int, default=100, metavar='N',
                        help='every how many mini-batches images will be saved in validation')
    parser.add_argument('--ckp_path', type=str, default=os.path.join(pathname,"checkpoints"), help='path to the training checkpoints' )
    parser.add_argument('--model_path', type=str, default=os.path.join(pathname,"model_checkpoints"), help='path to the training checkpoints' )
    
    parser.add_argument('--plane', type=str, default="axial", choices=["axial", "coronal", "sagittal"],
                        help="Which plane to train on (axial (default), coronal or sagittal)")
    
    parser.add_argument('--train_stdn', type = list, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.5, 8.0, 8.5, 9.0], help=" Number of noise levels (standard deviation) for which the network will be trained.") # [1.0,2.0,2.55,3.0,4.,5.0,5.10,6.0,7.,7.65,8.0,9.0,10.,11.0,12.0,12.75,13.0,14.0,15.0]
    parser.add_argument('--test_stdn', type = list, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.5, 8.0, 8.5, 9.0], help=" Number of noise levels (standard deviation) for testing.") # [2.55, 5.10, 7.65, 12.75]
    parser.add_argument('--is_train', type = bool, default = True, help=' True for training phase')
    parser.add_argument('--resume_training', type = bool, default = True, help=' True for resume training')
    parser.add_argument('--resume_with_lpips', type = bool, default = False, help=' True for using the previous best model based on the lpips metric for resuming training')
    parser.add_argument('--epochs_already_trained', type = int, default = 6, help='Total (net) number of epochs already trained for.')
    parser.add_argument('--is_mixup', type = bool, default = True, help=' mixup_data augmentation for training data')
    parser.add_argument('--rgb_range', type = int, default = 255, help='data range of the training images.') 
    parser.add_argument('--patch_size', type = int, default = 320, help='patch size for training. [x2-->64,x3-->42,x4-->32]')
    
    # optimizer
    parser.add_argument('--optimizer', default='AdamW',
                        choices=['AdamW', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
    # parser.add_argument('--momentum', default=0.99, type=float,
    #                     help='momentum')
    parser.add_argument('--weight_decay', default=1.2e-6, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='MultiStepLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR', 'OneCycleLR'])
    parser.add_argument('--min_lr', default=1e-8, type=float,
                        help='minimum learning rate')
    parser.add_argument('--max_lr', default=1e-5, type=float, help='maximum LR for One cycle policy')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--milestones', default='1', type=str)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

scaler = torch.cuda.amp.GradScaler()
def train(config, train_loader, model, criterion, optimizer, scheduler, epoch, writer, map_256, map_320, map_448):
    avg_meters = {'loss': AverageMeter()}
    
    
    model.train()

    pbar = tqdm(total=len(train_loader))
    n_iter = len(train_loader)
    # current_epoch = epoch
    
    
    for idx, data in enumerate(train_loader):
        current_iter = (epoch*n_iter)+idx #Current global iteration
        x, target, zoom, sigma = data['LR'], data['HR'], data['zoom'], data['sigma']
        input = x
        input = input.cuda()
        target = target.cuda()
        
        input = input.float()
        target = target.float()
        bs, c, h, w = target.shape
        template = torch.zeros(bs,h,w)
        loss=0
        mask = target > 0 
        mask = mask.float()
        
        # with torch.cuda.amp.autocast():
        #     output, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6 = model(input,zoom) #input shape is [BS, 7, 448, 448]
        #     loss_current = criterion(mask*output, target)
        with torch.cuda.amp.autocast():
            output, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6 = model(input,zoom) #input shape is [BS, 7, 448, 448]
        loss_current = criterion(mask*output, target)
        
        loss += loss_current.item()
        
        if idx%config['verbose_save_imgs_train']==0:
            writer.add_scalar("Training_Loss",loss, current_iter)
            
            alpha_1 = torch.abs(alpha_1)
            alpha_2 = torch.abs(alpha_2)
            alpha_3 = torch.abs(alpha_3)
            alpha_4 = torch.abs(alpha_4)
            alpha_5 = torch.abs(alpha_5)
            alpha_6 = torch.abs(alpha_6)
            
            writer.add_scalars('alphas', {'alpha_1':alpha_1.detach().cpu().numpy(),
                                            'alpha_2':alpha_2.detach().cpu().numpy(),
                                            'alpha_3':alpha_3.detach().cpu().numpy(),
                                            'alpha_4':alpha_4.detach().cpu().numpy(),
                                            'alpha_5':alpha_5.detach().cpu().numpy(),
                                            'alpha_6':alpha_6.detach().cpu().numpy()}, current_iter)
            
            writer.add_scalars('learning_rates', {'LR':np.asarray(scheduler.get_last_lr()).item()}, current_iter)
            input=input
            
            input = input[:,3].reshape(template.shape)*255.
            output = output.reshape(template.shape)*255.
            target = target.reshape(template.shape)*255.
            
            input = input.type(torch.uint8)
            output = output.type(torch.uint8)
            target = target.type(torch.uint8)
            
            target_img = target.cpu().numpy()
            input_img = input.cpu().numpy()
            output_img = output.cpu().numpy()
            
            if idx >= 0 and idx<10:
                image_index = '00000' + repr(idx)
            elif idx >= 10 and idx < 100:
                image_index = '0000' + repr(idx)
            elif idx >= 100 and idx < 1000:
                image_index = '000' + repr(idx)
            elif idx >= 1000 and idx < 10000:
                image_index = '00' + repr(idx)
            elif idx >= 10000 and idx < 100000:
                image_index = '0' + repr(idx)
            elif idx >= 100000 and idx < 1000000:
                image_index = repr(idx)
            
            for i in range(bs):
                save(os.path.join(config['save_imgs_path'],'img'+image_index+'_mb'+repr(i)+'_'+sigma[i]+'_gt'+'.png'), abs(target_img[i])) #mb means mini-batch position
                save(os.path.join(config['save_imgs_path'],'img'+image_index+'_mb'+repr(i)+'_'+sigma[i]+'_noisy'+'.png'), abs(input_img[i]))
                save(os.path.join(config['save_imgs_path'],'img'+image_index+'_mb'+repr(i)+'_'+sigma[i]+'_denoised'+'.png'), abs(output_img[i]))
                
        scaler.scale(loss_current).backward()
        # #gpu_usage_where(11)
        scaler.step(optimizer)
        # #gpu_usage_where(12)
        scaler.update()
        # #gpu_usage_where(13)
        scheduler.step()
        # #gpu_usage_where(14)
        # optimizer.step()
        #gpu_usage_where(15)
        optimizer.zero_grad()
        # torch.cuda.empty_cache()
        avg_meters['loss'].update(loss, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg)])


def validate(config, val_loader, model, criterion, epoch, writer, map_256, map_320, map_448):
    avg_meters = {'loss': AverageMeter()}
    
    
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        
        n_iter = len(val_loader)
        # for _, data in val_loader:
        for idx, data in enumerate(val_loader):
            current_iter = (epoch*n_iter)+idx #Current global iteration
            x, target, zoom, sigma = data['LR'], data['HR'], data['zoom'], data['sigma']
            # target = torch.unsqueeze(target,1)
            input = x
            input = input.cuda()
            target = target.cuda()
            input = input.float()
            target = target.float()
            
            with torch.cuda.amp.autocast():
                output, _, _, _, _, _, _ = model(input,zoom) #input shape is [BS, 7, 448, 448]
            bs, c, h, w = target.shape
            template = torch.zeros(bs,h,w)
            loss=0
            
            mask = target > 0 
            mask = mask.float()
            loss_current = criterion(mask*output, target)
            loss += loss_current
            
            if idx%config['verbose_save_imgs_val']==0:
                writer.add_scalar("Validation_Loss",loss, current_iter)
                
                input = input[:,3]
                input = input
                
                input = input.reshape(template.shape)*255.
                output = output.reshape(template.shape)*255.
                target = target.reshape(template.shape)*255.
                
                input = input.type(torch.uint8)
                output = output.type(torch.uint8)
                target = target.type(torch.uint8)
                
                target_img = target.cpu().numpy()
                input_img = input.cpu().numpy()
                output_img = output.cpu().numpy()
                
                if idx >= 0 and idx<10:
                    image_index = '00000' + repr(idx)
                elif idx >= 10 and idx < 100:
                    image_index = '0000' + repr(idx)
                elif idx >= 100 and idx < 1000:
                    image_index = '000' + repr(idx)
                elif idx >= 1000 and idx < 10000:
                    image_index = '00' + repr(idx)
                elif idx >= 10000 and idx < 100000:
                    image_index = '0' + repr(idx)
                elif idx >= 100000 and idx < 1000000:
                    image_index = repr(idx)
                
                for i in range(bs):
                    save(os.path.join(config['save_imgs_path_val'],'img'+image_index+'_mb'+repr(i)+'_'+sigma[i]+'_gt'+'.png'), abs(target_img[i])) #mb means mini-batch position
                    save(os.path.join(config['save_imgs_path_val'],'img'+image_index+'_mb'+repr(i)+'_'+sigma[i]+'_noisy'+'.png'), abs(input_img[i]))
                    save(os.path.join(config['save_imgs_path_val'],'img'+image_index+'_mb'+repr(i)+'_'+sigma[i]+'_denoised'+'.png'), abs(output_img[i]))
            
            avg_meters['loss'].update(loss.item(), input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg)])


def main():
    config = vars(parse_args())
    
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    if config['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.99))
    elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    else:
        raise NotImplementedError
    
        
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    # elif config['scheduler'] == 'OneCycleLR':
    #     scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=config['max_lr'], epochs = config['epochs'], steps_per_epoch=training_steps,
    #                                         anneal_strategy='cos', cycle_momentum=True, pct_start = 0.075, div_factor = 10, final_div_factor=100,
    #                                         verbose=False)
    else:
        raise NotImplementedError
    
    
    if config['resume_training']:
        model = model.cuda()
        last_ckp_path = get_last_ckp_path(config)
        print('Loading model from '+last_ckp_path)
        model, _, _, _, _ = load_ckp(last_ckp_path, model, optimizer, scheduler)
        #Update the new number of steps. Plus 2*epochs extra is to account for the missing 6 steps that happened when training for 3 epochs 
        # scheduler.total_steps = (config['epochs']*training_steps) + config['batch_size']*config['epochs']*4
        # writer_train = SummaryWriter(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'runs',config['name']+'_Training'), purge_step=last_epoch_resume*training_steps)
        # writer_validation = SummaryWriter(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'runs',config['name']+'_Validation'), purge_step=last_epoch_resume*validation_steps)
        # log = csv_to_odict('models/%s/log.csv' % config['name'])
        print('======CHECKPOINT LOADED======')
        
        # optimizer = optimizer.cuda()
        # scheduler = scheduler.cuda()
    last_epoch_resume=0
    
    print('=======DATASETS LOADER ============')
    transform_train = transforms.Compose([AugmentationPadImage(pad_size=0)])
    transform_test = None
    
    # Prepare and load data
    params_dataset_train1 = {'dataset_name': config['hdf5_name_train1'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                            'is_train':config['is_train'], 'rgb_range':config['rgb_range'], 'noise_std':config["train_stdn"]}
    params_dataset_train2 = {'dataset_name': config['hdf5_name_train2'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                            'is_train':config['is_train'], 'rgb_range':config['rgb_range'], 'noise_std':config["train_stdn"]}
    params_dataset_train3 = {'dataset_name': config['hdf5_name_train3'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                            'is_train':config['is_train'], 'rgb_range':config['rgb_range'], 'noise_std':config["train_stdn"]}
    params_dataset_train4 = {'dataset_name': config['hdf5_name_train4'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                            'is_train':config['is_train'], 'rgb_range':config['rgb_range'], 'noise_std':config["train_stdn"]}
    params_dataset_train5 = {'dataset_name': config['hdf5_name_train5'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                            'is_train':config['is_train'], 'rgb_range':config['rgb_range'], 'noise_std':config["train_stdn"]}
    params_dataset_train6 = {'dataset_name': config['hdf5_name_train6'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                            'is_train':config['is_train'], 'rgb_range':config['rgb_range'], 'noise_std':config["train_stdn"]}
    params_dataset_train7 = {'dataset_name': config['hdf5_name_train7'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                            'is_train':config['is_train'], 'rgb_range':config['rgb_range'], 'noise_std':config["train_stdn"]}
                            
    params_dataset_test1 = {'dataset_name': config['hdf5_name_val1'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                           'is_train':False, 'rgb_range':config['rgb_range'], 'noise_std':config["test_stdn"]}
    params_dataset_test2 = {'dataset_name': config['hdf5_name_val2'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                           'is_train':False, 'rgb_range':config['rgb_range'], 'noise_std':config["test_stdn"]}
    params_dataset_test3 = {'dataset_name': config['hdf5_name_val3'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                           'is_train':False, 'rgb_range':config['rgb_range'], 'noise_std':config["test_stdn"]}
    params_dataset_test4 = {'dataset_name': config['hdf5_name_val4'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                           'is_train':False, 'rgb_range':config['rgb_range'], 'noise_std':config["test_stdn"]}
    params_dataset_test5 = {'dataset_name': config['hdf5_name_val5'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                           'is_train':False, 'rgb_range':config['rgb_range'], 'noise_std':config["test_stdn"]}
    params_dataset_test6 = {'dataset_name': config['hdf5_name_val6'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                           'is_train':False, 'rgb_range':config['rgb_range'], 'noise_std':config["test_stdn"]}
    params_dataset_test7 = {'dataset_name': config['hdf5_name_val7'], 'plane': config['plane'], 'patch_size':config['patch_size'],
                           'is_train':False, 'rgb_range':config['rgb_range'], 'noise_std':config["test_stdn"]}
    
    # train for one epoch
    map_256 = get_var_map_size(size=256)
    map_256 = get_thick_slices_maponly(map_256)
    map_320 = get_var_map_size(size=320)
    map_320 = get_thick_slices_maponly(map_320)
    map_448 = get_var_map_size(size=448)
    map_448 = get_thick_slices_maponly(map_448)

    dataset_train_1 = AsegDatasetWithAugmentation(params_dataset_train1, map_256, map_320, map_448, transforms=transform_train)
    dataset_train_2 = AsegDatasetWithAugmentation(params_dataset_train2, map_256, map_320, map_448, transforms=transform_train)
    dataset_train_3 = AsegDatasetWithAugmentation(params_dataset_train3, map_256, map_320, map_448, transforms=transform_train)
    dataset_train_4 = AsegDatasetWithAugmentation(params_dataset_train4, map_256, map_320, map_448, transforms=transform_train)
    dataset_train_5 = AsegDatasetWithAugmentation(params_dataset_train5, map_256, map_320, map_448, transforms=transform_train)
    dataset_train_6 = AsegDatasetWithAugmentation(params_dataset_train6, map_256, map_320, map_448, transforms=transform_train)
    dataset_train_7 = AsegDatasetWithAugmentation(params_dataset_train7, map_256, map_320, map_448, transforms=transform_train)
    
    dataset_validation_1 = AsegDatasetWithAugmentation(params_dataset_test1, map_256, map_320, map_448, transforms=transform_test, is_val = True)
    dataset_validation_2 = AsegDatasetWithAugmentation(params_dataset_test2, map_256, map_320, map_448, transforms=transform_test, is_val = True)
    dataset_validation_3 = AsegDatasetWithAugmentation(params_dataset_test3, map_256, map_320, map_448, transforms=transform_test, is_val = True)
    dataset_validation_4 = AsegDatasetWithAugmentation(params_dataset_test4, map_256, map_320, map_448, transforms=transform_test, is_val = True)
    dataset_validation_5 = AsegDatasetWithAugmentation(params_dataset_test5, map_256, map_320, map_448, transforms=transform_test, is_val = True)
    dataset_validation_6 = AsegDatasetWithAugmentation(params_dataset_test6, map_256, map_320, map_448, transforms=transform_test, is_val = True)
    dataset_validation_7 = AsegDatasetWithAugmentation(params_dataset_test7, map_256, map_320, map_448, transforms=transform_test, is_val = True)

    dataset_train = ConcatDataset((dataset_train_1, dataset_train_2, dataset_train_3, dataset_train_4, dataset_train_5, dataset_train_6, dataset_train_7))
    dataset_validation = ConcatDataset((dataset_validation_1, dataset_validation_2, dataset_validation_3, dataset_validation_4, dataset_validation_5, dataset_validation_6, dataset_validation_7))
    
    a_len = dataset_train_1.__len__()
    b_len = a_len + dataset_train_2.__len__()
    c_len = b_len + dataset_train_3.__len__()
    d_len = c_len + dataset_train_4.__len__()
    e_len = d_len + dataset_train_5.__len__()
    f_len = e_len + dataset_train_6.__len__()
    g_len = f_len + dataset_train_7.__len__()
    
    a_indices = list(range(a_len))
    b_indices = list(range(a_len, b_len))
    c_indices = list(range(b_len, c_len))
    d_indices = list(range(c_len, d_len))
    e_indices = list(range(d_len, e_len))
    f_indices = list(range(e_len, f_len))
    g_indices = list(range(f_len, g_len))
    
    av_len = dataset_validation_1.__len__()
    bv_len = av_len + dataset_validation_2.__len__()
    cv_len = bv_len + dataset_validation_3.__len__()
    dv_len = cv_len + dataset_validation_4.__len__()
    ev_len = dv_len + dataset_validation_5.__len__()
    fv_len = ev_len + dataset_validation_6.__len__()
    gv_len = fv_len + dataset_validation_7.__len__()
    av_indices = list(range(av_len))
    bv_indices = list(range(av_len, bv_len))
    cv_indices = list(range(bv_len, cv_len))
    dv_indices = list(range(cv_len, dv_len))
    ev_indices = list(range(dv_len, ev_len))
    fv_indices = list(range(ev_len, fv_len))
    gv_indices = list(range(fv_len, gv_len))
    
    batch_sampler_train = MyBatchSampler(a_indices, b_indices, c_indices, d_indices, e_indices, f_indices, g_indices, config['batch_size'])
    batch_sampler_val = MyBatchSampler(av_indices, bv_indices, cv_indices, dv_indices, ev_indices, fv_indices, gv_indices, config['validation_batch_size'])
    
    train_loader = DataLoader(dataset_train, batch_sampler = batch_sampler_train)
    val_loader = DataLoader(dataset_validation, batch_sampler = batch_sampler_val)
    
    training_steps = len(train_loader)
    
    model = model.cuda()
    scheduler.total_steps = (config['epochs']*training_steps) + config['batch_size']*config['epochs']*4
    writer_train = SummaryWriter(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'runs',config['name']+'_Training'))
    writer_validation = SummaryWriter(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'runs',config['name']+'_Validation'))
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('val_loss', []),
    ])
    
    
    
    print('========== FINISHED LOADING DATASETS ==========')
    if not os.path.exists(config['save_imgs_path']):
        os.makedirs(config['save_imgs_path'])
        print('Created directory: '+ config['save_imgs_path'])
    
    if not os.path.exists(config['save_imgs_path_val']):
        os.makedirs(config['save_imgs_path_val'])
        print('Created directory: '+ config['save_imgs_path_val'])
        
    if not os.path.exists(config['ckp_path']):
        os.makedirs(config['ckp_path'])
        print('Created directory: '+ config['ckp_path'])
    
    if not os.path.exists(config['model_path']):
        os.makedirs(config['model_path'])
        print('Created directory: '+ config['model_path'])
              
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    # print('-' * 20)
    # for key in config:
    #     print('%s: %s' % (key, config[key]))
    # print('-' * 20)

    
    best_loss = 1
    trigger = 0
    
    for epoch in range(last_epoch_resume,config['epochs']):
        
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

     
        
        train_log = train(config, train_loader, model, criterion, optimizer, scheduler, epoch, writer_train, map_256, map_320, map_448)
        val_log_all = validate(config, val_loader, model, criterion, epoch, writer_validation, map_256, map_320, map_448)
        val_log = val_log_all
        
        
        
        if val_log['loss'] < best_loss:
            is_best = True
            best_loss = val_log['loss']
        else:
            is_best = False
        
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss
            }
            
        save_ckp(checkpoint, is_best, checkpoint_dir=config['ckp_path'], best_model_dir=config['model_path'])
        
        # if config['scheduler'] == 'CosineAnnealingLR':
        #     scheduler.step()
        # elif config['scheduler'] == 'ReduceLROnPlateau':
        #     scheduler.step(val_log['loss'])

        print('loss %.4f - val_loss %.4f '
              % (train_log['loss'], val_log['loss']))

        log['epoch'].append(epoch)
        log['lr'].append(np.asarray(scheduler.get_last_lr()).item())
        log['loss'].append(train_log['loss'])
        log['val_loss'].append(val_log['loss'])
        
        trigger += 1
        torch.cuda.empty_cache()
        copyf(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'runs',config['name']+'_Training'),
              os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'runs_ckp',config['name']+'_Training_epoch_'+str(epoch)+'ckp'))
        copyf(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'runs',config['name']+'_Validation'),
              os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'runs_ckp',config['name']+'_Validation_epoch_'+str(epoch)+'ckp'))

if __name__ == '__main__':
    main()
