# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:09:52 2022

@author: walte
"""


# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import h5py
import scipy.ndimage.morphology as morphology
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import sys
import random
import torch
from torchvision import transforms as trf
from skimage.measure import label
from torch.utils.data.dataset import Dataset
from .conform import is_conform, is_conform_itk, conform, check_affine_in_nifti, check_affine_in_nifti_itk, std_pos, conform_std, conform_mask, conform_std_mask, conform_keep_dims, conform_itk
from data_loader import common
from torch.utils.data.sampler import Sampler
from numpy import array
import os, tempfile
# from scipy.interpolate import Rbf 
from scipy.interpolate import RegularGridInterpolator as rgi
from PIL import Image
# from scipy.interpolate import griddata as rgi
# from scipy.interpolate import interpn as rgi
supported_output_file_formats = ['mgz', 'nii', 'nii.gz']
from datetime import datetime


##
# Helper Functions
##

def slice_img(orig, slice = 20):
    """
    Function that receives a 3D freesurfer image and returns a PIL image of the slice position "slice",
    which is the slicing position at the dim=2

    Parameters
    ----------
    orig : TYPE
        DESCRIPTION.
    slice : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    None.

    """
    sample = orig.get_fdata()
    sample = sample[:,:,slice]
    sample = (sample - sample.min()) / (sample.max() - sample.min())*255
    sample = sample.astype(np.uint8)
    sample = Image.fromarray(sample)
    return sample

def reorient_standard_RAS(img):
    ornt = np.array([[0, 1], [1, -1], [2, -1]])
    img_orient = img.as_reoriented(ornt) # re-orient the image
    return img

def chunk(indices, size):
    return torch.split(torch.tensor(indices), size)


# def add_rician_varying(x):
#     shape = x.shape
#     h = shape[0]
#     w = shape[1]
#     c = shape[2]
    
#     map1 = np.ones([3,3,3])
#     map1[1,1,1] = 3
#     nx, ny, nz = (3, 3, 3)
#     xi = np.linspace(1, nx, nx)
#     yi = np.linspace(1, ny, ny)
#     zi = np.linspace(1, nz, nz)
    
#     x1, y1, z1 = np.meshgrid(xi, yi, zi)
    
#     xi2 = np.linspace(1, nx, h)
#     yi2 = np.linspace(1, ny, w)
#     zi2 = np.linspace(1, nz, c)
    
#     x2, y2, z2 = np.meshgrid(xi2, yi2, zi2)
#     interp3 = rgi((y1,x1,z1), map1, np.array([x2,y2,z2]).T, method='cubic')
#     # interp3 = Rbf(x1, y1, z1, map1, x2, y2, z2 function="cubic")
#     final_map = interp3(array([x1, y1, z1]).T)
    
#     return final_map

def add_rician_varying(im):
    def f(x,y,z):
        if x==y and y==z:
            return 3
        else: 
            return 1
        shape = im.shape
        h = shape[0]
        w = shape[1]
        c = shape[2]
        
        nx, ny, nz = (3, 3, 3)
        xi = np.linspace(1, nx, nx)
        yi = np.linspace(1, ny, ny)
        zi = np.linspace(1, nz, nz)
        xg, yg, zg = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
        data = np.zeros((3,3,3))
        data = f(xg, yg, zg)
        interp3 = rgi((xi, yi, zi), data, method="cubic")
        
        xi2 = np.linspace(1, nx, h)
        yi2 = np.linspace(1, ny, w)
        zi2 = np.linspace(1, nz, c)
        
        x2, y2, z2 = np.meshgrid(xi2, yi2, zi2)
        final_map = interp3(array([x2, y2, z2]).T)
        
        return final_map
        
# Conform an MRI brain image to UCHAR, RAS orientation, and 1mm isotropic voxels
def load_and_conform_image(img_filename, interpol=1, logger=None, is_eval = False, conform_type = 2, intensity_rescaling = False):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param str conform_type: (0: Cubic shape of dims = max(img_filename.shape) and voxdim of minimum voxdim. 1: Cubic shape of dims 256^3. 2: Keep dimensions) 
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = nib.load(img_filename)
    orig = nib.funcs.squeeze_image(orig)
    max_orig = orig.get_fdata().max()
    min_orig = orig.get_fdata().min()
    zoom = orig.header.get_zooms()
    # orig = (orig - orig.min) / (orig.max-orig.min)
    ishape = orig.shape
    max_shape = max(ishape)
    if not is_conform(orig):

        if logger is not None:
            if conform_type == 0:
                logger.info('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
            else:
                logger.info('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        else:
            if conform_type == 0:
                print('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
            else:
                print('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        if len(orig.shape) > 3 and orig.shape[3] != 1:
            sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

        # Check affine if image is nifti image
        if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
            if not check_affine_in_nifti(orig, logger=logger):
                sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

        # conform
        if conform_type == 0:
            orig = conform(orig, order = interpol, conform_type = conform_type, intensity_rescaling = intensity_rescaling)
        elif conform_type == 1:
            orig = conform_std(orig, interpol, intensity_rescaling)
        elif conform_type == 2:
            orig = conform(orig, order = interpol, conform_type = conform_type, intensity_rescaling = intensity_rescaling)

    # Collect header and affine information
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asanyarray(orig.dataobj)
    if is_eval:
        return header_info, affine_info, orig, zoom, max_orig, min_orig
    else:
        return header_info, affine_info, orig
    
def load_and_conform_image_sitk(img_filename, interpol=1, logger=None, is_eval = False, conform_type = 2, intensity_rescaling = False):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param str conform_type: (0: Cubic shape of dims = max(img_filename.shape) and voxdim of minimum voxdim. 1: Cubic shape of dims 256^3. 2: Keep dimensions) 
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = sitk.ReadImage(img_filename)
    orig = sitk.DICOMOrient(orig, 'RAS')
    zoom = orig.GetSpacing()
    ishape = orig.GetSize()
    max_shape = max(ishape)
    # if len(ishape) == 4:
    #     extract_filter = sitk.ExtractImageFilter()
    #     extract_filter.SetSize([ishape[0], ishape[1], ishape[2]])
    #     orig = extract_filter.Execute(orig)
    
    orig_img = sitk.GetArrayFromImage(orig)
    orig_img = np.transpose(orig_img, (2, 1, 0))
    max_orig = orig_img.max()
    min_orig = orig_img.min()
    
    if not is_conform_itk(orig):
        if logger is not None:
            if conform_type == 0:
                logger.info('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
            else:
                logger.info('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        else:
            if conform_type == 0:
                print('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
            else:
                print('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        if len(orig_img.shape) > 3 and orig_img.shape[3] != 1:
            sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

        # Check affine if image is nifti image
        if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
            if not check_affine_in_nifti_itk(orig, logger=logger):
                sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")
            
            
            # # Split the filename into base and extension
            # base, ext = os.path.splitext(img_filename)
            # # If the extension is .gz, split again to get the .nii part
            # if ext == ".gz":
            #     base, ext = os.path.splitext(base)
            # # Add _temp to the base and join with the extension
            # now = datetime.now() # get current local datetime object
            # time_string = now.strftime("%Y_%m_%d_%H_%M_%S") # format as string
            # img_filename_temp = base + "_temp_" + time_string + ext
            # # Write the orig image object to the temp file
            # sitk.WriteImage(orig, img_filename_temp)
            # orig_temp = nib.load(img_filename_temp)
            
            # if not check_affine_in_nifti(orig_temp, logger=logger):
            #     sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")
            
            # # Delete the temp file
            # del orig_temp
            # try:
            #     os.remove(img_filename_temp)
            # except PermissionError:
            #     print("No se pudo eliminar el archivo temporal")
            
            
            # with tempfile.NamedTemporaryFile(delete=False) as tmp:
            #     img_filename_temp = tmp.name # get the name of the temporary file
            #     print(img_filename_temp) # prints something like '/tmp/tmpa_1di3b'
            #     # write the image using sitk.WriteImage()
            #     sitk.WriteImage(orig, img_filename_temp)
            #     orig_temp = nib.load(img_filename_temp)
            #     if not check_affine_in_nifti(orig_temp, logger=logger):
            #         sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")
            #     del orig_temp
            #     os.unlink(img_filename_temp)
            
            
            # if not check_affine_in_nifti(orig, logger=logger):
            #     sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

        # conform
        if conform_type == 0:
            orig = conform_itk(orig, order = interpol, conform_type = conform_type, intensity_rescaling = intensity_rescaling)
        elif conform_type == 1:
            orig = conform_std(orig, interpol, intensity_rescaling)
        elif conform_type == 2:
            orig = conform_itk(orig, order = interpol, conform_type = conform_type, intensity_rescaling = intensity_rescaling)
    orig_array = sitk.GetArrayFromImage(orig)
    orig_array = np.transpose(orig_array, (2, 1, 0))
    if is_eval:
        return orig, orig_array, zoom, max_orig, min_orig
    else:
        return orig, orig_array
    
def load_and_keep_dims(img_filename, interpol=1, logger=None, is_eval = False, conform_type = 0, intensity_rescaling = False):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param str conform_type: (0=min_vox_size+max_im_size, 1=std(1.0/256)) 
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = nib.load(img_filename)
    
    max_orig = orig.get_fdata().max()
    min_orig = orig.get_fdata().min()
    zoom = orig.header.get_zooms()
    # orig = (orig - orig.min) / (orig.max-orig.min)
    
    
    # orig = conform_keep_dims(orig, interpol, intensity_rescaling)

    # Collect header and affine information
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asanyarray(orig.dataobj)
    if is_eval:
        return header_info, affine_info, orig, zoom, max_orig, min_orig
    else:
        return header_info, affine_info, orig

def load_and_conform_image_mask(img_filename, mask_filename, interpol=1, logger=None, is_eval = False, conform_type = 0):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param str conform_type: (0=min_vox_size+max_im_size, 1=std(1.0/256)) 
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    
    orig = nib.load(img_filename)
    # affine = orig.affine
    # header = orig.header
    # orig = orig.get_data()
    # if len(orig.shape) == 4:
    #     orig = np.squeeze(orig)
    # orig = nib.MGHImage(orig, affine, header)
    zoom = orig.header.get_zooms()
    if len(zoom) == 4:
        zoom = zoom[:-1]
    mask = nib.load(mask_filename)
    ishape = orig.shape
    if len(ishape) == 4:
        ishape = ishape[:-1]
    max_shape = max(ishape)
    
    # if not is_conform(orig):

        # if logger is not None:
        #     if conform_type == 0:
        #         logger.info('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
        #     else:
        #         logger.info('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        # else:
        #     if conform_type == 0:
        #         print('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
        #     else:
        #         print('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        # if len(orig.shape) > 3 and orig.shape[3] != 1:
        #     sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

        # # Check affine if image is nifti image
        # if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
        #     if not check_affine_in_nifti(orig, logger=logger):
        #         sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

        # # conform
        # if conform_type == 0:
        #     orig = conform(orig, interpol)
        #     mask = conform_mask(mask, interpol)
        # elif conform_type == 1:
        #     orig = conform_std(orig, interpol)
        #     mask = conform_std_mask(mask, interpol)
    
    if logger is not None:
        if conform_type == 0:
            logger.info('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
        else:
            logger.info('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
    else:
        if conform_type == 0:
            print('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
        else:
            print('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
    if len(orig.shape) > 3 and orig.shape[3] != 1:
        sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

    # Check affine if image is nifti image
    if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
        if not check_affine_in_nifti(orig, logger=logger):
            sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

    # conform
    if conform_type == 0:
        orig = conform(orig, interpol)
        mask = conform_mask(mask, interpol)
    elif conform_type == 1:
        orig = conform_std(orig, interpol)
        mask = conform_std_mask(mask, interpol)

    # Collect header and affine information
    
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asanyarray(orig.dataobj)
    mask = np.asanyarray(mask.dataobj)
    
    mask = mask>0
    mask = mask.astype(np.uint8)
    
    orig = orig*mask
    
    if is_eval:
        return header_info, affine_info, orig, zoom
    else:
        return header_info, affine_info, orig
    
    

def load_image(img_filename, interpol=1, logger=None, is_eval = False):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = nib.load(img_filename)
    zoom = orig.header.get_zooms()
    orig = std_pos(orig)
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asanyarray(orig.dataobj)
    if is_eval:
        return header_info, affine_info, orig, zoom
    else:
        return header_info, affine_info, orig

def save_image(img_array, affine_info, header_info, save_as):
    """
    Save an image (nibabel MGHImage), according to the desired output file format.
    Supported formats are defined in supported_output_file_formats.

    :param numpy.ndarray img_array: an array containing image data
    :param numpy.ndarray affine_info: image affine information
    :param nibabel.freesurfer.mghformat.MGHHeader header_info: image header information
    :param str save_as: name under which to save prediction; this determines output file format

    :return None: saves predictions to save_as
    """

    assert any(save_as.endswith(file_ext) for file_ext in supported_output_file_formats), \
            'Output filename does not contain a supported file format (' + ', '.join(file_ext for file_ext in supported_output_file_formats) + ')!'

    mgh_img = None
    if save_as.endswith('mgz'):
        mgh_img = nib.MGHImage(img_array, affine_info, header_info)
    elif any(save_as.endswith(file_ext) for file_ext in ['nii', 'nii.gz']):
        mgh_img = nib.nifti1.Nifti1Pair(img_array, affine_info, header_info)

    if any(save_as.endswith(file_ext) for file_ext in ['mgz', 'nii']):
        nib.save(mgh_img, save_as)
    elif save_as.endswith('nii.gz'):
        ## For correct outputs, nii.gz files should be saved using the nifti1 sub-module's save():
        nib.nifti1.save(mgh_img, save_as)


# Transformation for mapping
def transform_axial(vol, sagittal2axial=True):
    """
    Function to transform volume into Axial axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2axial: transform from coronal to axial = True (default),
                               transform from axial to coronal = False
    :return: np.ndarray: transformed image volume
    """
    if sagittal2axial:
        vol = np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
        return vol
    else:
        vol = np.moveaxis(vol, [1, 2, 0], [0, 1, 2])
        return vol


def transform_coronal(vol, sagittal2coronal=True):
    """
    Function to transform volume into Sagittal axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2sagittal: transform from coronal to sagittal = True (default),
                                transform from sagittal to coronal = False
    :return: np.ndarray: transformed image volume
    """
    if sagittal2coronal:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])
    else:
        return np.moveaxis(vol, [1, 0, 2], [0, 1, 2])


# Thick slice generator (for eval) and blank slices filter (for training)
# def get_thick_slices(img_data, slice_thickness=3, anisotropic = True):
#     """
#     Function to extract thick slices from the image 
#     (feed slice_thickness preceeding and suceeding slices to network, 
#     denoise only middle one) Added a padding stage so all the images are the 
#     :param np.ndarray img_data: 3D MRI image read in with nibabel 
#     :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
#     :return: np.ndarray img_data_thick: image array containing the extracted slices
#     """
#     h, w, d = img_data.shape
#     img_data_pad = np.expand_dims(np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'),
#                                   axis=3)
#     img_data_thick = np.ndarray((h, w, d, 0), dtype=np.uint8)
    
#     for slice_idx in range(2 * slice_thickness + 1):
#         img_data_thick = np.append(img_data_thick, img_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)
#     return img_data_thick

def get_thick_slices(img_data, slice_thickness=3, anisotropic = True):
    """
    Function to extract thick slices from the image 
    (feed slice_thickness preceeding and suceeding slices to network, 
    denoise only middle one) Added a padding stage so all the images are the 
    :param np.ndarray img_data: 3D MRI image read in with nibabel 
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
    :param bool anisotropic: whether the image has different resolutions along different axes (default=True)
    :return: np.ndarray img_data_thick: image array containing the extracted slices
    """
    h, w, d = img_data.shape
    img_data_pad = np.expand_dims(np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'),
                                  axis=3)
    img_data_thick = np.ndarray((h, w, d, 0), dtype=np.uint8)
    
    if anisotropic:
        # If anisotropic is True, use only the middle slice and repeat it
        for slice_idx in range(2 * slice_thickness + 1):
            img_data_thick = np.append(img_data_thick, np.expand_dims(img_data, axis=3), axis=3)

    else:
        # If anisotropic is False, use consecutive slices as usual
        for slice_idx in range(2 * slice_thickness + 1):
            img_data_thick = np.append(img_data_thick, img_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)

    return img_data_thick

def get_noisy_pre_den_pairs(pre_denoised_image, orig):
    """
    Function to extract thick slices from the image 
    (feed slice_thickness preceeding and suceeding slices to network, 
    denoise only middle one) Added a padding stage so all the images are the 
    :param np.ndarray img_data: 3D MRI image read in with nibabel 
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
    :param bool anisotropic: whether the image has different resolutions along different axes (default=True)
    :return: np.ndarray img_data_thick: image array containing the extracted slices
    """
    img_data_pairs = np.concatenate ( (pre_denoised_image[..., None], orig[..., None]), axis=3) # array of size [448, 448, 448, 2]
    return img_data_pairs

def get_thick_slices_ms(img_data, max_size, orig_size, slice_thickness=3):
    """
    Function to extract thick slices from the image 
    (feed slice_thickness preceeding and suceeding slices to network, 
    denoise only middle one) Added a padding stage so all the images are the 
    :param np.ndarray img_data: 3D MRI image read in with nibabel 
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
    :return: np.ndarray img_data_thick: image array containing the extracted slices
    """
    h, w, d = img_data.shape
    img_data_pad = np.expand_dims(np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'),
                                  axis=3)
    img_data_thick = np.ndarray((h, w, d, 0), dtype=np.uint8)
    
    for slice_idx in range(2 * slice_thickness + 1):
        img_data_thick = np.append(img_data_thick, img_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)
        
    dif = max_size - orig_size
    to_pad = ((0,dif), (0,dif), (0,dif), (0,0))
    img_data_thick = np.pad(img_data_thick, pad_width=to_pad, mode='constant', constant_values = 0)
    return img_data_thick

def get_thick_slices_vmap(img_data, map_data, max_size, orig_size, slice_thickness=3):
    """
    Function to extract thick slices from the image 
    (feed slice_thickness preceeding and suceeding slices to network, 
    denoise only middle one) Added a padding stage so all the images are the 
    :param np.ndarray img_data: 3D MRI image read in with nibabel 
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
    :return: np.ndarray img_data_thick: image array containing the extracted slices
    """
    h, w, d = img_data.shape
    
    img_data_pad = np.expand_dims(np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'),axis=3)
    map_data_pad = np.expand_dims(np.pad(map_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'), axis=3)
    
    img_data_thick = np.ndarray((h, w, d, 0), dtype=np.uint8)
    map_data_thick = np.ndarray((h, w, d, 0), dtype=np.float64)
    
    for slice_idx in range(2 * slice_thickness + 1):
        img_data_thick = np.append(img_data_thick, img_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)
        map_data_thick = np.append(map_data_thick, map_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)
    return img_data_thick, map_data_thick 

def get_thick_slices_maponly(map_data, slice_thickness=3):
    """
    Function to extract thick slices from the image 
    (feed slice_thickness preceeding and suceeding slices to network, 
    denoise only middle one) Added a padding stage so all the images are the 
    :param np.ndarray img_data: 3D MRI image read in with nibabel 
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
    :return: np.ndarray img_data_thick: image array containing the extracted slices
    """
    h, w, d = map_data.shape
    
    map_data_pad = np.expand_dims(np.pad(map_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'), axis=3)
    
    map_data_thick = np.ndarray((h, w, d, 0), dtype=np.float64)
    
    for slice_idx in range(2 * slice_thickness + 1):
        map_data_thick = np.append(map_data_thick, map_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)
    return map_data_thick 

def filter_blank_slices_thick(img_vol, label_vol, weight_vol, threshold=50):
    """
    Function to filter blank slices from the volume using the label volume
    :param np.ndarray img_vol: orig image volume
    :param np.ndarray label_vol: label images (ground truth)
    :param np.ndarray weight_vol: weight corresponding to labels
    :param int threshold: threshold for number of pixels needed to keep slice (below = dropped)
    :return:
    :return: np.ndarray img_vol: filtered orig image volume
    :return: np.ndarray label_vol: filtered label images (ground truth)
    :return: np.ndarray weight_vol: filtered weight corresponding to labels
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = (np.sum(label_vol, axis=(0, 1)) > threshold)

    # Retain only slices with more than threshold labels/pixels
    img_vol = img_vol[:, :, select_slices, :]
    label_vol = label_vol[:, :, select_slices]
    weight_vol = weight_vol[:, :, select_slices]

    return img_vol, label_vol, weight_vol

def filter_blank_slices_thick_ms(img_vol, threshold=50):
    """
    Function to filter blank slices from the volume using the label volume
    :param np.ndarray img_vol: orig image volume
    :param np.ndarray label_vol: label images (ground truth)
    :param np.ndarray weight_vol: weight corresponding to labels
    :param int threshold: threshold for number of pixels needed to keep slice (below = dropped)
    :return:
    :return: np.ndarray img_vol: filtered orig image volume
    :return: np.ndarray label_vol: filtered label images (ground truth)
    :return: np.ndarray weight_vol: filtered weight corresponding to labels
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = (np.sum(img_vol, axis=(0, 1)) > threshold)

    # Retain only slices with more than threshold labels/pixels
    img_vol = img_vol[:, :, select_slices[:,1], :]

    return img_vol

def filter_blank_slices_thick_vmap(img_vol, threshold=50):
    """
    Function to filter blank slices from the volume using the label volume
    :param np.ndarray img_vol: orig image volume
    :param np.ndarray label_vol: label images (ground truth)
    :param np.ndarray weight_vol: weight corresponding to labels
    :param int threshold: threshold for number of pixels needed to keep slice (below = dropped)
    :return:
    :return: np.ndarray img_vol: filtered orig image volume
    :return: np.ndarray label_vol: filtered label images (ground truth)
    :return: np.ndarray weight_vol: filtered weight corresponding to labels
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = (np.sum(img_vol, axis=(0, 1)) > threshold)

    # Retain only slices with more than threshold labels/pixels
    img_vol = img_vol[:, :, select_slices[:,1], :]
    
    return img_vol, select_slices


# weight map generator
def create_weight_mask(mapped_aseg, max_weight=5, max_edge_weight=5):
    """
    Function to create weighted mask - with median frequency balancing and edge-weighting
    :param np.ndarray mapped_aseg: label space segmentation
    :param int max_weight: an upper bound on weight values
    :param int max_edge_weight: edge-weighting factor
    :return: np.ndarray weights_mask: generated weights mask
    """
    unique, counts = np.unique(mapped_aseg, return_counts=True)

    # Median Frequency Balancing
    class_wise_weights = np.median(counts) / counts
    class_wise_weights[class_wise_weights > max_weight] = max_weight
    (h, w, d) = mapped_aseg.shape

    weights_mask = np.reshape(class_wise_weights[mapped_aseg.ravel()], (h, w, d))

    # Gradient Weighting
    (gx, gy, gz) = np.gradient(mapped_aseg)
    grad_weight = max_edge_weight * np.asarray(np.power(np.power(gx, 2) + np.power(gy, 2) + np.power(gz, 2), 0.5) > 0,
                                               dtype='float')

    weights_mask += grad_weight

    return weights_mask


# class unknown filler (cortex)
def fill_unknown_labels_per_hemi(gt, unknown_label, cortex_stop):
    """
    Function to replace label 1000 (lh unknown) and 2000 (rh unknown) with closest class for each voxel.
    :param np.ndarray gt: ground truth segmentation with class unknown
    :param int unknown_label: class label for unknown (lh: 1000, rh: 2000)
    :param int cortex_stop: class label at which cortical labels of this hemi stop (lh: 2000, rh: 3000)
    :return: np.ndarray gt: ground truth segmentation with replaced unknown class labels
    """
    # Define shape of image and dilation element
    h, w, d = gt.shape
    struct1 = ndimage.generate_binary_structure(3, 2)

    # Get indices of unknown labels, dilate them to get closest sorrounding parcels
    unknown = gt == unknown_label
    unknown = (morphology.binary_dilation(unknown, struct1) ^ unknown)
    list_parcels = np.unique(gt[unknown])

    # Mask all subcortical structures (fill unknown with closest cortical parcels only)
    mask = (list_parcels > unknown_label) & (list_parcels < cortex_stop)
    list_parcels = list_parcels[mask]

    # For each closest parcel, blur label with gaussian filter (spread), append resulting blurred images
    blur_vals = np.ndarray((h, w, d, 0), dtype=float)

    for idx in range(len(list_parcels)):
        aseg_blur = filters.gaussian_filter(1000 * np.asarray(gt == list_parcels[idx], dtype=float), sigma=5)
        blur_vals = np.append(blur_vals, np.expand_dims(aseg_blur, axis=3), axis=3)

    # Get for each position parcel with maximum value after blurring (= closest parcel)
    unknown = np.argmax(blur_vals, axis=3)
    unknown = np.reshape(list_parcels[unknown.ravel()], (h, w, d))

    # Assign the determined closest parcel to the unknown class (case-by-case basis)
    mask = gt == unknown_label
    gt[mask] = unknown[mask]

    return gt


# Label mapping functions (to aparc (eval) and to label (train))
def map_label2aparc_aseg(mapped_aseg):
    """
    Function to perform look-up table mapping from label space to aparc.DKTatlas+aseg space
    :param np.ndarray mapped_aseg: label space segmentation
    :return: np.ndarray aseg: segmentation in aparc+aseg space
    """
    aseg = np.zeros_like(mapped_aseg)
    labels = np.array([0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                       15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                       46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                       77, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                       1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                       1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
                       2002, 2005, 2010, 2012, 2013, 2014, 2016, 2017, 2021, 2022, 2023,
                       2024, 2025, 2028])
    h, w, d = aseg.shape

    aseg = labels[mapped_aseg.ravel()]

    aseg = aseg.reshape((h, w, d))

    return aseg


def map_aparc_aseg2label(aseg, aseg_nocc=None):
    """
    Function to perform look-up table mapping of aparc.DKTatlas+aseg.mgz data to label space
    :param np.ndarray aseg: ground truth aparc+aseg
    :param None/np.ndarray aseg_nocc: ground truth aseg without corpus callosum segmentation
    :return: np.ndarray mapped_aseg: label space segmentation (coronal and axial)
    :return: np.ndarray mapped_aseg_sag: label space segmentation (sagittal)
    """
    aseg_temp = aseg.copy()
    aseg[aseg == 80] = 77  # Hypointensities Class
    aseg[aseg == 85] = 0  # Optic Chiasma to BKG
    aseg[aseg == 62] = 41  # Right Vessel to Right GM
    aseg[aseg == 30] = 2  # Left Vessel to Left GM
    aseg[aseg == 72] = 24  # 5th Ventricle to CSF

    # If corpus callosum is not removed yet, do it now
    if aseg_nocc is not None:
        cc_mask = (aseg >= 251) & (aseg <= 255)
        aseg[cc_mask] = aseg_nocc[cc_mask]

    aseg[aseg == 3] = 0  # Map Remaining Cortical labels to background
    aseg[aseg == 42] = 0

    # If ctx-unknowns are not filled yet, do it now
    if np.any(np.in1d([1000, 2000], aseg.ravel())):
        aseg = fill_unknown_labels_per_hemi(aseg, 1000, 2000)
        aseg = fill_unknown_labels_per_hemi(aseg, 2000, 3000)

    cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
    aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    # Preserve Cortical Labels
    aseg[aseg_temp == 2014] = 2014
    aseg[aseg_temp == 2028] = 2028
    aseg[aseg_temp == 2012] = 2012
    aseg[aseg_temp == 2016] = 2016
    aseg[aseg_temp == 2002] = 2002
    aseg[aseg_temp == 2023] = 2023
    aseg[aseg_temp == 2017] = 2017
    aseg[aseg_temp == 2024] = 2024
    aseg[aseg_temp == 2010] = 2010
    aseg[aseg_temp == 2013] = 2013
    aseg[aseg_temp == 2025] = 2025
    aseg[aseg_temp == 2022] = 2022
    aseg[aseg_temp == 2021] = 2021
    aseg[aseg_temp == 2005] = 2005

    labels = np.array([0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                       15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                       46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                       77, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                       1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                       1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
                       2002, 2005, 2010, 2012, 2013, 2014, 2016, 2017, 2021, 2022, 2023,
                       2024, 2025, 2028])

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels) + 1, dtype='int')
    for idx, value in enumerate(labels):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Coronal, Axial

    mapped_aseg = lut_aseg.ravel()[aseg.ravel()]

    mapped_aseg = mapped_aseg.reshape((h, w, d))

    # Map Sagittal Labels
    aseg[aseg == 2] = 41
    aseg[aseg == 3] = 42
    aseg[aseg == 4] = 43
    aseg[aseg == 5] = 44
    aseg[aseg == 7] = 46
    aseg[aseg == 8] = 47
    aseg[aseg == 10] = 49
    aseg[aseg == 11] = 50
    aseg[aseg == 12] = 51
    aseg[aseg == 13] = 52
    aseg[aseg == 17] = 53
    aseg[aseg == 18] = 54
    aseg[aseg == 26] = 58
    aseg[aseg == 28] = 60
    aseg[aseg == 31] = 63

    cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
    aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    labels_sag = np.array([0, 14, 15, 16, 24, 41, 43, 44, 46, 47, 49,
                           50, 51, 52, 53, 54, 58, 60, 63, 77, 1002,
                           1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014,
                           1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025,
                           1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035])

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels_sag) + 1, dtype='int')
    for idx, value in enumerate(labels_sag):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Sagittal

    mapped_aseg_sag = lut_aseg.ravel()[aseg.ravel()]

    mapped_aseg_sag = mapped_aseg_sag.reshape((h, w, d))

    return mapped_aseg, mapped_aseg_sag


def sagittal_coronal_remap_lookup(x):
    """
    Dictionary mapping to convert left labels to corresponding right labels for aseg
    :param int x: label to look up
    :return: dict: left-to-right aseg label mapping dict
    """
    return {
        2: 41,
        3: 42,
        4: 43,
        5: 44,
        7: 46,
        8: 47,
        10: 49,
        11: 50,
        12: 51,
        13: 52,
        17: 53,
        18: 54,
        26: 58,
        28: 60,
        31: 63,
        }[x]


def map_prediction_sagittal2full(prediction_sag, num_classes=79):
    """
    Function to remap the prediction on the sagittal network to full label space used by coronal and axial networks
    (full aparc.DKTatlas+aseg.mgz)
    :param np.ndarray prediction_sag: sagittal prediction (labels)
    :param int num_classes: number of classes (96 for full classes, 79 for hemi split)
    :return: np.ndarray prediction_full: Remapped prediction
    """
    if num_classes == 96:
        idx_list = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 14, 15, 4, 16,
                               17, 18, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 20, 21, 22,
                               23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], dtype=np.int16)

    else:
        idx_list = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 14, 15, 4, 16,
                               17, 18, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 20, 22, 27,
                               29, 30, 31, 33, 34, 38, 39, 40, 41, 42, 45], dtype=np.int16)

    prediction_full = prediction_sag[:, idx_list, :, :]
    return prediction_full


# Clean up and class separation
def bbox_3d(img):
    """
    Function to extract the three-dimensional bounding box coordinates.
    :param np.ndarray img: mri image
    :return: float rmin
    :return: float rmax
    :return: float cmin
    :return: float cmax
    :return: float zmin
    :return: float zmax
    """

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def get_largest_cc(segmentation):
    """
    Function to find largest connected component of segmentation.
    :param np.ndarray segmentation: segmentation
    :return: np.ndarray largest_cc: largest connected component of the segmentation array
    """
    labels = label(segmentation, connectivity=3, background=0)

    bincount = np.bincount(labels.flat)
    background = np.argmax(bincount)
    bincount[background] = -1

    largest_cc = labels == np.argmax(bincount)

    return largest_cc


# Class Operator for image loading (orig only)
class OrigDataThickSlices(Dataset):
    """
    Class to load a given image and segmentation and prepare it
    for network training.
    """
    def __init__(self, img_filename, orig, plane='Axial', slice_thickness=3, transforms=None, anisotropic = True):

        try:
            self.img_filename = img_filename
            self.plane = plane
            self.slice_thickness = slice_thickness
            self.anisotropic = anisotropic
            orig.astype(np.float64)
            orig = (orig-orig.min())/(orig.max() - orig.min())
            # Transform Data as needed
            if plane == 'Sagittal':
                #Volume dims are (Sagittal, Coronal, Axial), therefore, 
                #we do not need to change the axis .
                print('Loading Sagittal')

            elif plane == 'Axial':
                #Volume dims are (Sagittal, Coronal, Axial), therefore, we have 
                #to change the Axial dimension to the first dimension.
                orig = transform_axial(orig)
                print('Loading Axial')

            else:
                #Volume dims are (Sagittal, Coronal, Axial), therefore, we have 
                #to change the Coronal dimension to the first dimension.
                orig = transform_coronal(orig)
                print('Loading Coronal.')

            # Create Thick Slices
            orig_thick = get_thick_slices(orig, self.slice_thickness, self.anisotropic)
            
            # # Make 4D
            # orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
            self.images = orig_thick

            self.count = self.images.shape[0]

            self.transforms = transforms

            print("Successfully loaded Image from {}".format(img_filename))

        except Exception as e:
            print("Loading failed. {}".format(e))

    def __getitem__(self, index):

        img = self.images[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return {'image': img}

    def __len__(self):
        return self.count
    

# class OrigDataTwoSlices(Dataset):
#     """
#     Class to load a given image and segmentation and prepare it
#     for network training.
#     """
#     def __init__(self, pre_denoised_image, orig, plane='Axial', transforms=None, anisotropic = True):

#         try:
#             self.plane = plane
#             self.anisotropic = anisotropic
#             orig.astype(np.float64)
#             # orig = (orig-orig.min())/(orig.max() - orig.min())
#             # pre_denoised_image.astype(np.float64)
#             # pre_denoised_image = (pre_denoised_image - pre_denoised_image.min()) / (pre_denoised_image.max() - pre_denoised_image.min())
#             # Transform Data as needed
#             if plane == 'Sagittal':
#                 orig = transform_sagittal(orig)
#                 pre_denoised_image = transform_sagittal(pre_denoised_image)
#                 print('Loading Sagittal')

#             elif plane == 'Axial':
#                 orig = transform_axial(orig)
#                 pre_denoised_image = transform_axial(pre_denoised_image)
#                 print('Loading Axial')

#             else:
#                 print('Loading Coronal.')

#             # Create Thick Slices
#             orig_pairs = get_noisy_pre_den_pairs(pre_denoised_image, orig)
            
#             # Make 4D
#             orig_pairs = np.transpose(orig_pairs, (2, 0, 1, 3))
#             self.images = orig_pairs

#             self.count = self.images.shape[0]

#             self.transforms = transforms

#             print("Successfully loaded pre-denoised + original image pairs ")

#         except Exception as e:
#             print("Loading failed. {}".format(e))

#     def __getitem__(self, index):

#         img = self.images[index]

#         if self.transforms is not None:
#             img = self.transforms(img)

#         return {'image': img}

#     def __len__(self):
#         return self.count


##
# Dataset loading (for training)
##

# Operator to load hdf5-file for training
class AsegDatasetWithAugmentation(Dataset):
    """
    Class for loading aseg file with augmentations (transforms)
    """
    def __init__(self, params, map_256, map_320, map_448, transforms=None, is_val = False):

        # Load the h5 file and save it to the dataset
        try:
            self.params = params

            # Open file in reading mode
            with h5py.File(self.params['dataset_name'], "r") as hf:
                self.images = np.array(hf.get('orig_dataset')[:])
                self.orig_zooms = np.array(hf.get('orig_zooms')[:])
                self.field_localizer = np.array(hf.get('field_localizer')[:])
                self.subjects = np.array(hf.get("subject")[:])
                self.noise_std = params['noise_std']
                self.patch_size = params['patch_size']
                self.map_256 = map_256
                self.map_320 = map_320
                self.map_448 = map_448
                
                self.is_val = is_val

            self.count = self.images.shape[0]
            self.transforms = transforms

            print("Successfully loaded {} with plane: {}".format(params["dataset_name"], params["plane"]))

        except Exception as e:
            print("Loading failed: {}".format(e))

    def get_subject_names(self):
        return self.subjects
    
    def _get_patch(self, img, localizer):
        map_size = max(img.shape)
        if map_size == 256:
            vmap = self.map_256
        elif map_size ==320:
            vmap = self.map_320
        elif map_size == 448:
            vmap = self.map_448
            
        vmap = vmap[int(localizer)]
        
        img = np.moveaxis(img,(0,1,2),(2,1,0))
        maxi = img.max()
        mini = img.min()
        img = (img - mini) / (maxi-mini)
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=1)
        hr = img[3]
        sigma = random.choice(self.noise_std)
        noise = 'R' + str(sigma)
        # LR_size = self.patch_size
        #NOISE ADDING
        lr = common.add_noise(img, vmap, noise)
        
        #AUGMENTATION
        # randcrop=trf.RandomCrop(LR_size)
        randflip=trf.RandomHorizontalFlip()
        if self.is_val == True:
            randrot=trf.RandomRotation([0,0])
        else:
            randrot=trf.RandomRotation([-5,5])
        lrhr = self.concatenate_Thick_GT(lr, hr)
        # lrhr = randcrop(lrhr)
        lrhr = randflip(lrhr)
        lrhr = randrot(lrhr)
        lr, hr = self.unconcatenate_Thick_GT(lrhr)
        
        return lr, hr, str(sigma)
    
    def __getitem__(self, index):

        img = self.images[index]
        zoom = self.orig_zooms[index]
        localizer = self.field_localizer[index]
        if self.transforms is not None:
            tx_sample = self.transforms({'img': img})
            img = tx_sample['img']
            #NEW PATCH ADDITION
            
        lr, hr, sigma = self._get_patch(img, localizer)
            
                
            # img, label = common.np2Tensor([lr, hr], self.rgb_range)

        return {'LR': lr, 'HR': hr, 'zoom':zoom, 'sigma':sigma}

    def __len__(self):
        return self.count
    
    

    def concatenate_Thick_GT(self, lr, hr):
        #Here, HR is ok
        hr = torch.from_numpy(hr)
        hr2 = hr.float()
        #hr converts to torch correctly
        hr2 = torch.unsqueeze(hr,0)
        lr = torch.from_numpy(lr)
        # lr = torch.unsqueeze(lr,0)
        lrhr = torch.cat([hr2,lr],0) #first 7 arrays of dim 0 are lr, last is hr
        return lrhr
    
    def unconcatenate_Thick_GT(self, lrhr):
        lr = lrhr[1:8] #This means from index 0 to index 7, but not including index 7
        hr = lrhr[0]
        return lr, hr
    
class MyBatchSampler(Sampler):
    def __init__(self, a_indices, b_indices, c_indices, d_indices, e_indices, f_indices, g_indices,  batch_size): 
        self.a_indices = a_indices
        self.b_indices = b_indices
        self.c_indices = c_indices
        self.d_indices = d_indices
        self.e_indices = e_indices
        self.f_indices = f_indices
        self.g_indices = g_indices
        self.batch_size = batch_size
    
    def __iter__(self):
        random.shuffle(self.a_indices)
        random.shuffle(self.b_indices)
        random.shuffle(self.c_indices)
        random.shuffle(self.d_indices)
        random.shuffle(self.e_indices)
        random.shuffle(self.f_indices)
        random.shuffle(self.g_indices)
        a_batches = chunk(self.a_indices, self.batch_size)
        b_batches = chunk(self.b_indices, self.batch_size)
        c_batches = chunk(self.c_indices, self.batch_size)
        d_batches = chunk(self.d_indices, self.batch_size)
        e_batches = chunk(self.e_indices, self.batch_size)
        f_batches = chunk(self.f_indices, self.batch_size)
        g_batches = chunk(self.g_indices, self.batch_size)
        all_batches = list(a_batches + b_batches + c_batches + d_batches + e_batches + f_batches + g_batches)
        all_batches = [batch.tolist() for batch in all_batches]
        random.shuffle(all_batches)
        return iter(all_batches)
    
    def __len__(self):
        return (len(self.a_indices) + len(self.b_indices) + len(self.c_indices) + len(self.d_indices) + len(self.e_indices) + len(self.f_indices) + len(self.g_indices)) // self.batch_size