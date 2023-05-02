
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
from data_loader.load_neuroimaging_data_final import load_and_conform_image, load_and_keep_dims
from data_loader.checkpoints import load_model, get_best_ckp_path
# from data_loader.conform import deconform
from pathlib import Path
import torch.backends.cudnn as cudnn
import yaml

HELPTEXT = """
Script to generate denoised.mgz using Deep Learning. /n

Dependencies:

    albumentations==1.3.0
    h5py==3.7.0
    imageio==2.19.3
    lpips==0.1.4
    matplotlib==3.5.2
    nibabel==5.1.0
    numpy==1.21.5
    opencv_python==4.7.0.72
    pandas==1.4.4
    Pillow==9.5.0
    PyYAML==6.0
    scikit_image==0.19.2
    scikit_learn==1.0.2
    scipy==1.9.1
    torch==1.13.1
    torchvision==0.14.1
    tqdm==4.64.1
    XlsxWriter==3.0.3
    torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

Original Author: Walter Adame Gonzalez
VANDAlab - Douglas Mental Health University Institute
PI - Mahsa Dadar, PhD., MSc.

Date: May-02-2023
"""

def gzip_this(in_file):
    import gzip
    import os
    in_data = open(in_file, "rb").read() # read the file as bytes
    out_gz = in_file + ".gz" # the name of the compressed file
    gzf = gzip.open(out_gz, "wb") # open the compressed file in write mode
    gzf.write(in_data) # write the data to the compressed file
    gzf.close() # close the file
    
    # If you want to delete the original file after the gzip is done:
    os.unlink(in_file)

def options_parse():
    """
    Command line option parser
    """
    pathname = os.path.dirname(sys.argv[0])
    parser = argparse.ArgumentParser(description=HELPTEXT, epilog='$Id: dccr-net, v 1.0 2023-02-05$')

    # 1. Directory information (where to read from, where to write to)
    parser.add_argument('--i_dir', '--input_directory', dest='input', default = os.path.join(os.path.dirname(sys.argv[0]),'input_MRI'),   help='path to directory of input volume(s).')
    parser.add_argument('--csv_file', '--csv_file', help="CSV-file with directories to process", default=None)
    parser.add_argument('--o_dir', '--output_directory', dest='output', default = os.path.join(os.path.dirname(sys.argv[0]),'denoised_MRI'),
                        help='path to output directory. Will be created if it does not already exist')
    
    parser.add_argument('--keep_dims', default = False, 
                        help = "If True then it will keep the original image size (HxWxC) and voxel size (v1xv2xv3)."
                        "If False then it will reshape the image to (M,M,M) where M=max(H,W,C) and voxel size to (v,v,v) where v=min(v1,v2,v3)"
                        "Default is False")
    parser.add_argument('--intensity_range_mode', default = 0, 
                        help = "Voxel intensity range for the generated images. 0 is for [0-255]. 1 is for using original intensity range (not recommended)."
                        "1 is for [0-1]")
    # 2. Options for the MRI volumes (name of in and output, order of interpolation if not conformed)
    parser.add_argument('--in_name', '--input_name', dest='iname', help='name of file to process. Default: orig.mgz',
                        default="D:/sub-01/av_highres_nlin.nii")
    
    parser.add_argument('--suffix', help='suffix of the denoised file. Default: orig.mgz',
                        default="dccr2")
    
    aux_parser = parser.parse_args()
    iname = aux_parser.iname
    fname = Path(iname)
    basename = os.path.join(fname.parent, fname.stem)
    ext = fname.suffix
    if ext == ".gz":
        fname2 = Path(basename)
        basename = os.path.join(fname2.parent, fname2.stem)
        ext = fname2.suffix + ext
        
    suffix = aux_parser.suffix
    keep_dims = aux_parser.keep_dims
    
    parser.add_argument('--out_name', '--output_name', dest='oname',
                        default = basename + "_" + suffix,
                        help='denoised filename without extension')
    
    if not keep_dims:
        parser.add_argument('--in_name_new', '--input_name_new', dest='iname_new', help='Conformed MRI filename',
                        default = basename + "_orig")
        
    
    parser.add_argument('--ext', default = ext, help = "Output file extension. By default is the same as the input extension")
    parser.add_argument('--order', dest='order', type=int, default=1,
                        help="order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)")
    
    # 6. Clean up and GPU/CPU options (disable cuda, change batchsize)
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

def run_network(img_filename, zoom, orig_data, denoised_image, plane, params_model, model, logger, args):
    """
    

    Parameters
    ----------
    img_filename : str
        Full path to the neuroimaging file.
    zoom : tuple
        Tuple of floats corresponding to the original voxel sizes retrieved from nibabel.header.get_zooms().
    orig_data : numpy array
        3D array containing the image to be denoised.
    denoised_image : torch tensor
        Tensor that will contain the denoised image.
    plane : str
        Plane to be used for denoising. "Axial", "Coronal" or "Sagittal".
    params_model : dict
        Values to be used to create the DataLoader variable that will be the input to the network.
    model : pytorch model (nn.Module)
        DCCR model containing the architecture to the network.
    logger : logger file
        DESCRIPTION.
    args : argument parser object
        Will contain all the arguments from the argument parser.

    Returns
    -------
    denoised_image : torch tensor
        Tensor that contains the denoised image.

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
                denoised_image[:, start_index:start_index + temp.shape[1], :, :] += torch.mul(temp.cpu(), 1)
                start_index += temp.shape[1]

            elif plane == "Coronal":
                temp = temp.permute(2, 3, 0, 1)
                denoised_image[:, :, start_index:start_index + temp.shape[2], :] += torch.mul(temp.cpu(), 1)
                start_index += temp.shape[2]

            else:
                temp = temp.permute(0, 3, 2, 1)
                denoised_image[start_index:start_index + temp.shape[0], :, :, :] += torch.mul(temp.cpu(), 1)
                start_index += temp.shape[0]

            logger.info("--->Batch {} {} Testing Done.".format(batch_idx, plane))

    return denoised_image


def resunetcnn(img_filename, save_as, save_as_new_orig, logger, args):
    """
    

    Parameters
    ----------
    img_filename : str
        Full path to the input of the image (image to be denoised).
    save_as : str
        Full output filename (without the extension -e.g. without the ".mnc"-) where the denoised image will be written to.
    save_as_new_orig : str
        Full filename (without the file extension) where the new input image will be written to. This will be only used in case that the --keep_dims flag is False.
    logger : logger file
        File containing the log to the pipeline.
    args : argument parser object
        Contains the input arguments to the script.

    Returns
    -------
    None.

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
    
    if not options.keep_dims:
        if conform_type < 2:
            header_info, affine_info, orig_data, orig_zoom, max_orig, min_orig = load_and_conform_image(os.path.join(options.input,img_filename), interpol=1, logger=logger, is_eval = True, conform_type = conform_type)
        else:
            header_info, affine_info, orig_data, orig_zoom, max_orig, min_orig = load_and_conform_image(os.path.join(options.input,img_filename), interpol=1, logger=logger, is_eval = True, conform_type = 2)
    else:
        header_info, affine_info, orig_data, orig_zoom, max_orig, min_orig = load_and_keep_dims(os.path.join(options.input,img_filename), interpol=1, logger=logger, is_eval = True, conform_type = conform_type)
    h, w, c = orig_data.shape
    max_orig2 = orig_data.max()
    min_orig2 = orig_data.min()
    orig_data = (orig_data - min_orig2) / (max_orig2 - min_orig2)
    
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
    
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")
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
    denoised_image = torch.zeros((h, w, c, num_planes_clean), dtype=torch.float)

    # Axial Prediction
    start = time.time()
    denoised_image = run_network(img_filename, orig_zoom,
                            orig_data_noisy, denoised_image, "Axial",
                            params_model, model, logger, args)
    
    logger.info("Axial View Tested in {:0.4f} seconds".format(time.time() - start))

    # Coronal Prediction
    start = time.time()
    denoised_image = run_network(img_filename, orig_zoom,
                            orig_data, denoised_image, "Coronal",
                            params_model, model, logger, args)
    logger.info("Coronal View Tested in {:0.4f} seconds".format(time.time() - start))

    # # Sagittal Prediction
    start = time.time()
    denoised_image = run_network(img_filename, orig_zoom,
                            orig_data, denoised_image, "Sagittal",
                            params_model, model, logger, args)
    denoised_image = torch.squeeze(denoised_image)
    logger.info("Sagittal View Tested in {:0.4f} seconds".format(time.time() - start))
    
    denoised_image /= 3.0
    denoised_image = denoised_image.clamp(0.0,1.0)
    
    if args.intensity_range_mode == 0:
        denoised_image = denoised_image * 255.
        orig_data = orig_data * 255.
    elif args.intensity_range_mode == 2:
        denoised_image = (denoised_image*(max_orig - min_orig)) + min_orig
        orig_data = (orig_data*(max_orig - min_orig)) + min_orig
    elif int(args.intensity_range_mode) <= -1 or int(args.intensity_range_mode) >= 3:
        print("Warning: intensity_range_mode not valid. Valid values are only 0 (0-255), 1 (orig_min - orig_max), and 2 (0-1)")
        print("Storing output with intensity range mode 0 (0-255) ....")
        denoised_image = denoised_image * 255.
        orig_data = orig_data * 255.
    # Saving image
    denoised_image = torch.squeeze(denoised_image)
    header_info.set_data_dtype(np.single)
    
    ext = options.ext
    if options.ext == ".mgz" or options.ext == ".mgh":
        den_img = nib.MGHImage(denoised_image, affine_info, header_info)
    elif options.ext == ".nii" or options.ext == ".mnc": 
        den_img = nib.Nifti1Image(denoised_image, affine_info, header_info)
        ext = ".nii"
    elif options.ext == ".nii.gz":
        den_img = nib.Nifti1Image(denoised_image, affine_info, header_info)
        ext = ".nii"
    # elif options.ext == ".mnc":
    #     den_img = nib.Minc2Image(denoised_image, affine_info, header_info)
    else:
        print("WARNING: Invalid extension: " + options.ext)
        print("Attempting storing the denoised image using .nii ...")
        den_img = nib.Nifti1Image(denoised_image, affine_info, header_info)
        ext = ".nii"
    # if conform_type == 2:
    #     den_img = deconform(den_img, orig_shape, orig_zoom, orig_max_intens = orig_max_intens, orig_min_intens = orig_min_intens)
    
    den_fname = save_as + ext
    nib.save(den_img,den_fname)
    
    if options.ext == ".nii.gz":
        gzip_this(save_as + ext)
        den_fname = den_fname + ".gz"
        
    logger.info("Saved clean MRI to {}".format(den_fname))
    
    if not options.keep_dims:
        if options.ext == ".mgz" or options.ext == ".mgh":
            orig_image_resampled = nib.MGHImage(orig_data, affine_info, header_info)
        elif options.ext == ".nii" or options.ext == ".mnc": 
            orig_image_resampled = nib.Nifti1Image(orig_data, affine_info, header_info)
        elif options.ext == ".nii.gz":
            orig_image_resampled = nib.Nifti1Image(orig_data, affine_info, header_info)
            ext = ".nii"
        # elif options.ext == ".mnc":
        #     orig_image_resampled = nib.Minc2Image(orig_data, affine_info, header_info)
        else:
            print("WARNING: Invalid extension: " + options.ext)
            print("Attempting storing the denoised image using .nii ...")
            orig_image_resampled = nib.Nifti1Image(orig_data, affine_info, header_info)
        
        orig_fname = save_as_new_orig + ext
        if options.ext == ".nii.gz":
            gzip_this(orig_fname)
            orig_fname = orig_fname + ".gz"
        logger.info("Saving original MRI rescaled  {}".format(orig_fname))
        nib.save(orig_image_resampled, save_as_new_orig + ext)
    
    logger.info("Total processing time: {:0.4f} seconds.".format(time.time() - start_total))
    
if __name__ == "__main__":

    options = options_parse()
    if options.ext == ".nii" or options.ext == ".nii.gz" or options.ext == ".mgh" or options.ext == ".mgz" or options.ext == ".mnc":
        valid_extensions = True
    else:
        valid_extensions = False
        
    # Set up the logger
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    
    if valid_extensions:
        if not options.keep_dims:
            resunetcnn(options.iname, os.path.join(options.oname), os.path.join(options.output,options.iname_new) , logger, options)
        else:
            resunetcnn(options.iname, os.path.join(options.output,options.oname), "_" , logger, options)
    else:
        print("Invalid input file extension. Valid extensions are .nii, .nii.gz, .mgh, .mgz, .mnc")
        sys.exit(0)

    sys.exit(0)
