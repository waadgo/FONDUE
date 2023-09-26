import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# IMPORTS
import argparse
import nibabel as nib
import numpy as np
import time
import sys
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import SimpleITK as sitk
from data_loader.load_neuroimaging_data_final import OrigDataThickSlices
from data_loader.load_neuroimaging_data_final import load_and_conform_image
from data_loader.checkpoints import load_model, get_best_ckp_path
from pathlib import Path
import torch.backends.cudnn as cudnn
import yaml
import importlib

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
def is_anisotropic(z1, z2, z3):
    # find the largest value
    largest = max(z1, z2, z3)
    # determine which axis has the largest value
    if largest == z1:
        irr_pos = "Sagittal"
    elif largest == z2:
        irr_pos = "Coronal"
    else:
        irr_pos = "Axial"
    # find the smallest value
    smallest = min(z1, z2, z3)
    # compare the largest and smallest values
    if largest >= 2 * smallest:
        print("WARNING: Voxel size is at least twice as large in the largest dimension than in the smallest dimension. Will perform denoising only using the "+irr_pos+" plane.")
        return True, irr_pos
    else:
        return False, 0

def options_parse():
    """
    Command line option parser
    """
    pathname = os.path.dirname(sys.argv[0])
    parser = argparse.ArgumentParser(description=HELPTEXT, epilog='$Id: dccr-net, v 1.0 2023-02-05$')

    parser.add_argument('--keep_dims', type=bool, default = True, 
                        help = "If True then it will keep the original image size (HxWxC) and voxel size (v1xv2xv3)."
                        "If False then it will reshape the image to (M,M,M) where M=max(H,W,C) and voxel size to (v,v,v) where v=min(v1,v2,v3)"
                        "Default is False")
    parser.add_argument('--intensity_range_mode', type=int, default = 2,
                        help = "Voxel intensity range for the generated images. 0 is for [0-255]. 2 is for using original intensity range (not recommended)."
                        "1 is for [0-1]")
    parser.add_argument('--save_new_input', type=bool, default = False, 
                        help = "If true, it will save the intensity-rescaled/reshaped/re-oriented (or a combination of these) version that was computed before producing the denoised image"
                        "Default is False")
    # 2. Options for the MRI volumes (name of in and output, order of interpolation if not conformed)
    parser.add_argument('--robust_rescale_input', type=bool, default = True,
                        help = "Perform rescaling of input intensity between 0-255 using histogram robust rescaling. If False rescaling will be simple rescaling using maximum and minimum values.")
    parser.add_argument('--in_name', '--input_name', dest='iname', type=str, help='name of file to process. Default: orig.mgz',
                        default="C:/Users/walte/Dropbox/fondue_sr_tests/DH2236_20230721_T1w_07_mm.nii")
    parser.add_argument('--csv_file', type=str, help='File that contains the list of volumes to denoise. Should be in csv format, one line per string',
                        default = None)
    parser.add_argument('--name', type = str, default='fondue_a',
                        choices=['fondue_a', 'fondue_b', 'fondue_b1', 'fondue_b2'],
                        help='model name')
    
    aux_parser = parser.parse_args()
    iname = aux_parser.iname
    fname = Path(iname)
    basename = os.path.join(fname.parent, fname.stem)
    ext = fname.suffix
    model_name = aux_parser.name
    if ext == ".gz":
        fname2 = Path(basename)
        basename = os.path.join(fname2.parent, fname2.stem)
        ext = fname2.suffix + ext
        
    keep_dims = aux_parser.keep_dims
    irm = aux_parser.intensity_range_mode
    rri = aux_parser.robust_rescale_input
    
    if keep_dims:
        suffix_2 = "_kd1"
    else:
        suffix_2 = "_kd0"
    if irm == 0:
        suffix_3 = "_irm0"
    elif irm == 2:
        suffix_3 = "_irm2"
    else:
        suffix_3 = "_irm1"
    if rri:
        suffix_4 = "_rri1"
    else:
        suffix_4 = "_rri0"
        
    settings_suffix = suffix_2 + suffix_3 + suffix_4
            
    
    parser.add_argument('--suffix', type=str, help='suffix of the denoised file. Default: orig.mgz',
                        default = model_name + settings_suffix)
    
    aux_parser = parser.parse_args()
    suffix = aux_parser.suffix
    parser.add_argument('--out_name', '--output_name', dest='oname', type=str,
                        default = basename + "_" + suffix,
                        help='denoised filename without extension')
    
    
    parser.add_argument('--in_name_new', '--input_name_new', dest='iname_new',
                        type=str, help='Conformed MRI filename',
                    default = basename + "_orig")
        
    
    parser.add_argument('--ext', type=str, default = ".nii.gz", help = "Output file extension. By default is the same as the input extension")
    parser.add_argument('--order', type=str, dest='order', default=1,
                        help="order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)")
    
    # 6. Clean up and GPU/CPU options (disable cuda, change batchsize)
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference. Default: 1")
    parser.add_argument('--conform_type', type=int, default=2, 
                        help="0: conform to regular volume size to the largest available and voxel size to smallest available || 1: conform to 256^3 and 1mm isotropic voxel size || 2: do not change image size or shape")
    parser.add_argument('--model_path', type=str, default=os.path.join(pathname,"model_checkpoints",model_name+".pt"), help='path to the training checkpoints' )
    sel_option = parser.parse_args()

    if sel_option.iname is None and sel_option.csv_file is None and not sel_option.simple_run:
        parser.print_help(sys.stderr)
        sys.exit('----------------------------/nERROR: Please specify data directory or input volume/n')
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

def run_network(img_filename, zoom, orig_data, denoised_image, plane, params_model, model, logger, args, anisotropic):
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
    test_dataset = OrigDataThickSlices(img_filename, orig_data, plane=plane, anisotropic = anisotropic) #this is going to rotate the axis acordingly with the plane

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
            if params_model["use_cuda"]:
                images_batch = images_batch.cuda()
            
            with torch.cuda.amp.autocast():
                temp,_,_,_,_,_,_ = model(images_batch, zoom)
            
            output = temp
            temp = output
            if plane == "Axial":
                temp = temp.permute(2, 3, 0, 1)
                denoised_image[:, :, start_index:start_index + temp.shape[2], :] += torch.mul(temp.cpu(), 1)
                start_index += temp.shape[2]

            elif plane == "Coronal":
                temp = temp.permute(3, 0, 2, 1)
                denoised_image[:, start_index:start_index + temp.shape[1], :, :] += torch.mul(temp.cpu(), 1)
                start_index += temp.shape[1]

            else:
                temp = temp.permute(0, 2, 3, 1)
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
    options = options_parse()
    archs = importlib.import_module("archs." + options.name)
    #LOADING CONFIGURATION FOR NETWORK
    
    with open('models/%s/config.yml' % options.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    
    cudnn.benchmark = True
    logger.info("Reading volume {}".format(img_filename))
    header_info, affine_info, orig_data, orig_zoom, max_orig, min_orig = load_and_conform_image(os.path.join(img_filename), interpol=1, logger=logger, is_eval = True, conform_type = 2)
  
    # CHECKING FOR MULTI-FRAMES IN LAST DIM
    ishape = sitk.ReadImage(img_filename)
    ishape = ishape.GetSize()
    h, w, c = orig_data.shape
    z1, z2, z3 = orig_zoom
    anisotropic, irr_pos = is_anisotropic(z1, z2, z3)
    max_orig2 = orig_data.max()
    min_orig2 = orig_data.min()
    orig_data = (orig_data - min_orig2) / (max_orig2 - min_orig2)
    
    orig_data_noisy = orig_data
    
    # Select the model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    
    # Put it onto the GPU or CPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
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
    if (anisotropic and irr_pos == "Axial") or not anisotropic: 
        start = time.time()
        denoised_image = run_network(img_filename, orig_zoom,
                                orig_data_noisy, denoised_image, "Axial",
                                params_model, model, logger, args, anisotropic)
        
        logger.info("Axial View Tested in {:0.4f} seconds".format(time.time() - start))

    # Coronal Prediction
    if (anisotropic and irr_pos == "Coronal") or not anisotropic:
        start = time.time()
        denoised_image = run_network(img_filename, orig_zoom,
                                orig_data, denoised_image, "Coronal",
                                params_model, model, logger, args, anisotropic)
        logger.info("Coronal View Tested in {:0.4f} seconds".format(time.time() - start))

    # # Sagittal Prediction
    if (anisotropic and irr_pos == "Sagittal") or not anisotropic:
        start = time.time()
        denoised_image = run_network(img_filename, orig_zoom,
                                orig_data, denoised_image, "Sagittal",
                                params_model, model, logger, args, anisotropic)
        logger.info("Sagittal View Tested in {:0.4f} seconds".format(time.time() - start))
        
    
    if not anisotropic:
        denoised_image /= 3.0
    
    if int(args.intensity_range_mode) == 0:
        denoised_image = denoised_image * 255.
        orig_data = orig_data * 255.
    elif int(args.intensity_range_mode) == 2:
        denoised_image = (denoised_image*(max_orig - min_orig)) + min_orig
        orig_data = (orig_data*(max_orig - min_orig)) + min_orig
    elif int(args.intensity_range_mode) < 0 or int(args.intensity_range_mode) > 2:
        print("Warning: intensity_range_mode not valid. Valid values are only 0 (0-255), 1 (orig_min - orig_max), and 2 (0-1)")
        print("Storing output with intensity range mode 0 (0-255) ....")
        denoised_image = denoised_image * 255.
        orig_data = orig_data * 255.
    # Saving image
    if options.ext == ".mgz" or options.ext == ".mgh":
        den_img = nib.MGHImage(denoised_image, affine_info, header_info)
    elif options.ext == ".nii" or options.ext == ".mnc": 
        den_img = nib.Nifti1Image(denoised_image, affine_info, header_info)
        ext = ".nii"
    elif options.ext == ".nii.gz":
        den_img = nib.Nifti1Image(denoised_image, affine_info, header_info)
        ext = ".nii"
    else:
        print("WARNING: Invalid extension: " + options.ext)
        print("Attempting storing the denoised image using .nii ...")
        den_img = nib.Nifti1Image(denoised_image, affine_info, header_info)
        ext = ".nii"
    if options.ext == ".nii.gz":
        is_gz = True
        options.ext = ".nii"
    den_fname = save_as + options.ext
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
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    if options.csv_file is not None:
        import csv
        # with open(options.csv_file, newline='', encoding='utf-8-sig') as f:
        with open(options.csv_file, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                iname = row[0]
                fname = Path(iname)
                basename = os.path.join(fname.parent, fname.stem)
                output_dir = os.path.join(fname.parent.parent,"FONDUE_B")
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                suffix = options.suffix
                oname = os.path.join(output_dir,fname.stem + "_" + suffix)
                iname_new = basename + "_orig" 
                if not os.path.exists(oname+options.ext):
                    if not options.keep_dims or options.save_new_input:
                        resunetcnn(iname, os.path.join(oname), os.path.join(iname_new) , logger, options)
                    else:
                        resunetcnn(iname, os.path.join(oname), "_" , logger, options)
                else:
                    print("Skipping file ... "+oname+options.ext)
    else:
        if options.ext == ".nii" or options.ext == ".nii.gz" or options.ext == ".mgh" or options.ext == ".mgz" or options.ext == ".mnc":
            valid_extensions = True
        else:
            valid_extensions = False
            
        # Set up the logger
        
        
        if valid_extensions:
            if not options.keep_dims or options.save_new_input:
                resunetcnn(options.iname, os.path.join(options.oname), os.path.join(options.iname_new) , logger, options)
            else:
                resunetcnn(options.iname, os.path.join(options.oname), "_" , logger, options)
        else:
            print("Invalid input file extension. Valid extensions are .nii, .nii.gz, .mgh, .mgz, .mnc")
            sys.exit(0)
    
        sys.exit(0)
