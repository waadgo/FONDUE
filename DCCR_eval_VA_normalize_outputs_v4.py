
# IMPORTS
import argparse
import nibabel as nib
import numpy as np
import time
import sys
import logging
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import archs
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from data_loader.load_neuroimaging_data_final import OrigDataThickSlices
from data_loader.load_neuroimaging_data_final import map_prediction_sagittal2full
from data_loader.load_neuroimaging_data_final import load_and_conform_image
from data_loader.load_neuroimaging_data_final import load_image
from data_loader.checkpoints import load_model, get_best_ckp_path
from data_loader.conform import deconform

import torch.backends.cudnn as cudnn
import yaml

HELPTEXT = """
Script to generate denoised.mgz using Deep Learning. /n

Dependencies:

    Torch 
    Torchvision
    Skimage
    Numpy
    Matplotlib
    h5py
    scipy
    Python 3.5
    Nibabel (to read and write neuroimaging data, http://nipy.org/nibabel/)


Original Author: Leonie Henschel

Date: Mar-12-2019

"""


def options_parse():
    """
    Command line option parser
    """
    pathname = os.path.dirname(sys.argv[0])
    parser = argparse.ArgumentParser(description=HELPTEXT, epilog='$Id: fast_surfer_cnn, v 1.0 2019/09/30$')

    # 1. Directory information (where to read from, where to write to)
    parser.add_argument('--i_dir', '--input_directory', dest='input', default = os.path.join(os.path.dirname(sys.argv[0]),'input_MRI'),   help='path to directory of input volume(s).')
    parser.add_argument('--csv_file', '--csv_file', help="CSV-file with directories to process", default=None)
    parser.add_argument('--o_dir', '--output_directory', dest='output', default = os.path.join(os.path.dirname(sys.argv[0]),'denoised_MRI'),
                        help='path to output directory. Will be created if it does not already exist')
    
    # 2. Options for the MRI volumes (name of in and output, order of interpolation if not conformed)
    # parser.add_argument('--in_name', '--input_name', dest='iname', help='name of file to process. Default: orig.mgz',
    #                     default='MEMPRAGE_010_r1.mgz')
    
    parser.add_argument('--in_name', '--input_name', dest='iname', help='name of file to process. Default: orig.mgz',
                        default="C:/Users/walte/Dropbox/Validation_dataset/tests/which_dccr_is_15/noisy_001.mgz")
    # parser.add_argument('--in_name', '--input_name', dest='iname', help='name of file to process. Default: orig.mgz',
    #                     default=os.path.join("D:/OneDrive - McGill University/Documents/Masters-McGill/DCCR-Net/Subjects/Gio_Validation/MEMPRAGE_015_aligned/meanrep001_aligned.nii"))
    # parser.add_argument('--out_name', '--output_name', dest='oname',
    #                     default=os.path.join("C:/Denoising","Dataset","nlpca","meanrep001_DCCR14_2_C_2PReLU_ckp4_view_agg"),
    #                     help='name under which segmentation will be saved. Default: aparc.DKTatlas+aseg.deep.mgz. '
    #                          'If a separate subfolder is desired (e.g. FS conform, add it to the name: '
    #                          'mri/aparc.DKTatlas+aseg.deep.mgz)'
    
    parser.add_argument('--out_name', '--output_name', dest='oname',
                        default=os.path.join("C:/Users/walte/Dropbox/Validation_dataset/tests/which_dccr_is_15/dccr_001_fine_low_high_noise"),
                        help='name under which segmentation will be saved. Default: aparc.DKTatlas+aseg.deep.mgz. '
                             'If a separate subfolder is desired (e.g. FS conform, add it to the name: '
                             'mri/aparc.DKTatlas+aseg.deep.mgz)')
    
    
    
    # parser.add_argument('--in_name_new', '--input_name_new', dest='iname_new', help='name of file to process. Default: orig.mgz',
    #                     default=os.path.join("C:/Denoising","Dataset","nlpca","tp1tomean1to6_echo_1_orig_"))
    parser.add_argument('--in_name_new', '--input_name_new', dest='iname_new', help='name of file to process. Default: orig.mgz',
                        default=os.path.join("C:/Users/walte/Dropbox/Validation_dataset/tests/abide_x3/dccr_v15_fine_low_high_noise_orig"))
    
    
    
    parser.add_argument('--order', dest='order', type=int, default=1,
                        help="order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)")

    # 3. Options for log-file and search-tag
    parser.add_argument('--t', '--tag', dest='search_tag', default="*",
                        help='Search tag to process only certain subjects. If a single image should be analyzed, '
                             'set the tag with its id. Default: processes all.')
    parser.add_argument('--log', dest='logfile', help='name of log-file. Default: deep-seg.log',
                        default='deep-seg.log')

    # 4. Pre-trained weights
    parser.add_argument('--network_sagittal_path', dest='network_sagittal_path',
                        help="path to pre-trained weights of sagittal network",
                        default='./checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
    parser.add_argument('--network_coronal_path', dest='network_coronal_path',
                        help="pre-trained weights of coronal network",
                        default='./models/DCCR-Net_Axial/model.pth')
    parser.add_argument('--network_axial_path', dest='network_axial_path',
                        help="pre-trained weights of axial network",
                        default=os.path.join(os.getcwd(),"models","DCCR-Net_Axial_v13_v4","model_epoch_5.pth"))

    # 5. Options for model parameters setup (only change if model training was changed)
    parser.add_argument('--num_filters', type=int, default=256,
                        help='Filter dimensions for DenseNet (all layers same). Default=64')
    parser.add_argument('--num_classes_ax_cor', type=int, default=79,
                        help='Number of classes to predict in axial and coronal net, including background. Default=79')
    parser.add_argument('--num_classes_sag', type=int, default=51,
                        help='Number of classes to predict in sagittal net, including background. Default=51')
    parser.add_argument('--num_channels', type=int, default=7,
                        help='Number of input channels. Default=7 (thick slices)')
    parser.add_argument('--kernel_height', type=int, default=5, help='Height of Kernel (Default 5)')
    parser.add_argument('--kernel_width', type=int, default=5, help='Width of Kernel (Default 5)')
    parser.add_argument('--stride', type=int, default=1, help="Stride during convolution (Default 1)")
    parser.add_argument('--stride_pool', type=int, default=2, help="Stride during pooling (Default 2)")
    parser.add_argument('--pool', type=int, default=2, help='Size of pooling filter (Default 2)')

    # 6. Clean up and GPU/CPU options (disable cuda, change batchsize)
    parser.add_argument('--clean', dest='cleanup', help="Flag to clean up segmentation", action='store_true')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference. Default: 1")
    parser.add_argument('--conform_type', type=int, default=0, 
                        help="0: conform to regular volume size to the largest available and voxel size to smallest available || 1: conform to 256^3 and 1mm isotropic voxel size || 2: conform to 880^3 voxels and 0.25mm voxel size and bring back to original size after denoising")
    parser.add_argument('--simple_run', action='store_true', default=True,
                        help='Simplified run: only analyse one given image specified by --in_name (output: --out_name). '
                             'Need to specify absolute path to both --in_name and --out_name if this option is chosen.')
    parser.add_argument('--name', default='DCCR-Net_v15_1_fine_low_high_noise',
                        help='model name')
    parser.add_argument('--ckp_path', type=str, default=os.path.join(pathname,"checkpoints"), help='path to the training checkpoints' )
    parser.add_argument('--model_path', type=str, default=os.path.join(pathname,"model_checkpoints"), help='path to the training checkpoints' )
    sel_option = parser.parse_args()

    if sel_option.input is None and sel_option.csv_file is None and not sel_option.simple_run:
        parser.print_help(sys.stderr)
        sys.exit('----------------------------/nERROR: Please specify data directory or input volume/n')

    if sel_option.output is None and not sel_option.simple_run:
        parser.print_help(sys.stderr)
        sys.exit('----------------------------/nERROR: Please specify data output directory '
                 '(can be same as input directory)/n')

    return sel_option

def add_noise(x, noise='.'):
        noise_type = noise[0]
        noise_value = float(noise[1:])/100
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            # noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = abs(x.astype(np.float64) + noises.astype(np.float64))
        return x_noise

def run_network(img_filename, zoom, orig_data, prediction_probability, plane, ckpts, params_model, model, logger, args):
    """
    Inference run for single network on a given image.

    :param str img_filename: name of image file
    :param np.ndarray orig_data: image data
    :param torch.tensor prediction_probability: default tensor to hold prediction probabilities
    :param str plane: Which plane to predict (Axial, Sagittal, Coronal)
    :param str ckpts: Path to pretrained weights of network
    :param dict params_model: parameters to set up model (includes device, use_cuda, model_parallel, batch_size)
    :param torch.nn.Module model: Model to use for prediction
    :param logging.logger logger: Logging instance info messages will be written to
    :return:
    """
    # Set up DataLoader
    test_dataset = OrigDataThickSlices(img_filename, orig_data, plane=plane) #this is going to rotate the axis acordingly with the plane

    test_data_loader = DataLoader(dataset=test_dataset, shuffle=False,
                                  batch_size=params_model["batch_size"])
    if params_model["use_cuda"]:
        model = model.cuda()
    best_ckp_path = get_best_ckp_path(args)
    print('Loading checkpoint from '+best_ckp_path)
    model = load_model(best_ckp_path, model)
    print('======CHECKPOINT LOADED======')
            
    model.eval()

    logger.info("{} model loaded.".format(plane))
    with torch.no_grad():

        start_index = 0
        for batch_idx, sample_batch in enumerate(test_data_loader):
            
            images_batch = Variable(sample_batch["image"])
            images_batch = images_batch.permute(0, 3, 1, 2) #Reshape from [BS, input_h, input_w, thickslice_size] to [BS, thickslice_size, input_h, input_w]
            images_batch = images_batch.float() #Transform to float to fit the network
            # images_batch /= 255.
            
            if params_model["use_cuda"]:
                images_batch = images_batch.cuda()
            
            temp,_,_,_,_,_,_ = model(images_batch, zoom)
            
            #IF DEEP SUPERVISIONlab
            
            
            output = temp
            # output = (output[0] + output[1])/2.
            temp = torch.squeeze(output)
            temp = output
            
            
            # del output, outputs
            if plane == "Axial":
                temp = temp.permute(3, 0, 2, 1)
                prediction_probability[:, start_index:start_index + temp.shape[1], :, :] += torch.mul(temp.cpu(), 1)
                start_index += temp.shape[1]

            elif plane == "Coronal":
                temp = temp.permute(2, 3, 0, 1)
                prediction_probability[:, :, start_index:start_index + temp.shape[2], :] += torch.mul(temp.cpu(), 1)
                start_index += temp.shape[2]

            else:
                temp = temp.permute(0, 3, 2, 1)
                prediction_probability[start_index:start_index + temp.shape[0], :, :, :] += torch.mul(temp.cpu(), 1)
                start_index += temp.shape[0]

            logger.info("--->Batch {} {} Testing Done.".format(batch_idx, plane))

    return prediction_probability


def resunetcnn(img_filename, save_as, save_as_new_orig, logger, args):
    """
    Cortical parcellation of single image with FastSurferCNN.

    :param str img_filename: name of image file
    :param parser.Argparse args: Arguments (passed via command line) to set up networks
            * args.network_sagittal_path: path to sagittal checkpoint (stored pretrained network)
            * args.network_coronal_path: path to coronal checkpoint (stored pretrained network)
            * args.network_axial_path: path to axial checkpoint (stored pretrained network)
            * args.cleanup: Whether to clean up the segmentation (medial filter on certain labels)
            * args.no_cuda: Whether to use CUDA (GPU) or not (CPU)
            * args.batch_size: Input batch size for inference (Default=8)
            * args.num_classes_ax_cor: Number of classes to predict in axial/coronal net (Default=79)
            * args.num_classes_sag: Number of classes to predict in sagittal net (Default=51)
            * args.num_channels: Number of input channels (Default=7, thick slices)
            * args.num_filters: Number of filter dimensions for DenseNet (Default=64)
            * args.kernel_height and args.kernel_width: Height and width of Kernel (Default=5)
            * args.stride: Stride during convolution (Default=1)
            * args.stride_pool: Stride during pooling (Default=2)
            * args.pool: Size of pooling filter (Default=2)
    :param logging.logger logger: Logging instance info messages will be written to
    :param str save_as: name under which to save prediction.

    :return None: saves prediction to save_as
    """
    start_total = time.time()
    
    #LOADING CONFIGURATION FOR NETWORK
    options = options_parse()
    with open('models/%s/config.yml' % options.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    
    cudnn.benchmark = True
    

    logger.info("Reading volume {}".format(img_filename))

    # header_info, affine_info, orig_data, orig_zoom, orig_shape, orig_max_intens, orig_min_intens = load_and_conform_image(os.path.join(options.input,img_filename), interpol=1, logger=logger, is_eval = True, conform_type = 2)
    conform_type = options.conform_type
    if conform_type < 2:
        header_info, affine_info, orig_data, orig_zoom = load_and_conform_image(os.path.join(options.input,img_filename), interpol=1, logger=logger, is_eval = True, conform_type = conform_type)
    else:
        header_info, affine_info, orig_data, orig_zoom, orig_shape, orig_max_intens, orig_min_intens = load_and_conform_image(os.path.join(options.input,img_filename), interpol=1, logger=logger, is_eval = True, conform_type = 2)

    h, w, c = orig_data.shape
    max_orig = orig_data.max()
    min_orig = orig_data.min()
    orig_data = (orig_data - min_orig) / (max_orig - min_orig)
    
    orig_data_noisy = orig_data
    
    # Select the model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    # model = FastSurferCNN(params_network)
    
    # Put it onto the GPU or CPU
    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = True
    
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    logger.info("Cuda available: {}, # Available GPUS: {}, "
                "Cuda user disabled (--no_cuda flag): {}, "
                "--> Using device: {}".format(torch.cuda.is_available(),
                                              torch.cuda.device_count(),
                                              args.no_cuda, device))

    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_parallel = True
    else:
        model_parallel = False

    model.to(device)
    
    model.eval()

    params_model = {'device': device, "use_cuda": use_cuda, "batch_size": args.batch_size,
                    "model_parallel": model_parallel} #modifications needed?

    # Set up tensor to hold probabilities
    num_planes_clean = 1
    pred_prob = torch.zeros((h, w, c, num_planes_clean), dtype=torch.float)

    # Axial Prediction
    start = time.time()
    pred_prob = run_network(img_filename, orig_zoom,
                            orig_data_noisy, pred_prob, "Axial",
                            args.network_axial_path,
                            params_model, model, logger, args)
    # pred_prob_axial = torch.squeeze(pred_prob_axial)
    # pred_prob_axial = torch.moveaxis(pred_prob_axial, (0,1,2),(2,1,0))
    # pred_prob = pred_prob*max_orig
    
    logger.info("Axial View Tested in {:0.4f} seconds".format(time.time() - start))

    # Coronal Prediction
    start = time.time()
    pred_prob = run_network(img_filename, orig_zoom,
                            orig_data, pred_prob, "Coronal",
                            args.network_axial_path,
                            params_model, model, logger, args)
    # pred_prob_coronal = torch.squeeze(pred_prob_coronal)
    # pred_prob_coronal = torch.moveaxis(pred_prob_coronal, (0,1,2),(2,1,0))

    logger.info("Coronal View Tested in {:0.4f} seconds".format(time.time() - start))

    # # Sagittal Prediction
    start = time.time()
    pred_prob = run_network(img_filename, orig_zoom,
                            orig_data, pred_prob, "Sagittal",
                            args.network_axial_path,
                            params_model, model, logger, args)
    pred_prob = torch.squeeze(pred_prob)
    # pred_prob_sagittal = torch.moveaxis(pred_prob_sagittal, (0,1,2),(2,1,0))

    logger.info("Sagittal View Tested in {:0.4f} seconds".format(time.time() - start))
    
    pred_prob /= 3.0
    
    pred_prob = pred_prob.clamp(0.0,1.0)
    # Saving image
    pred_prob = torch.squeeze(pred_prob) #reshape from [448, 448, 448, 1] to [448, 448, 448]
    # pred_prob = (pred_prob - pred_prob.min()) / (pred_prob.max() - pred_prob.min())
    header_info.set_data_dtype(np.uint8)
    
    mapped_aseg_img = nib.MGHImage(255*pred_prob, affine_info, header_info)
    if conform_type == 2:
        mapped_aseg_img = deconform(mapped_aseg_img, orig_shape, orig_zoom, orig_max_intens = orig_max_intens, orig_min_intens = orig_min_intens)
        
    mapped_aseg_img.to_filename(save_as+'.mgz')
    
    
    # orig_data = (orig_data - orig_data.min()) / (orig_data.max() - orig_data.min())
    gt_image = nib.MGHImage(orig_data*255, affine_info, header_info)
    gt_image.to_filename(save_as_new_orig+'gt.mgz')
    
    logger.info("Saving clean MRI to {}".format(save_as))
    logger.info("Saving original MRI rescaled between 0 and 255 to {}".format(save_as_new_orig))
    logger.info("Total processing time: {:0.4f} seconds.".format(time.time() - start_total))
    # print(model)


if __name__ == "__main__":

    options = options_parse()

    # Set up the logger
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    resunetcnn(options.iname, os.path.join(options.output,options.oname), os.path.join(options.output,options.iname_new) , logger, options)

    sys.exit(0)
