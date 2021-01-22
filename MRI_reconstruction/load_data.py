"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import os
import h5py
from matplotlib import pyplot as plt
import fastmri
from fastmri.data import transforms as T
import torch
from fastmri.data.subsample import RandomMaskFunc
import functions


def get_files(file_dir):
    file_path = []
    for fname in sorted(os.listdir(file_dir)):
        subject_path = os.path.join(file_dir, fname)
        file_path.append(subject_path)
    print('there are ' + str(len(file_path)) + ' files')
    return file_path


def load_dataset(file_name):
    dataset = h5py.File(file_name, 'r')
    volume_kspace = dataset['kspace'][()]
    # print(volume_kspace.shape)
    slice_num = volume_kspace.shape[0]
    return volume_kspace, slice_num


file_dir = '/Users/huyuyang/singlecoil_val'
'''''''''
file_path = get_files(file_dir)
total_kspace, slices_num = load_dataset(file_path[0])
image_list = []
slice_kspace_tensor_list = []
for i in range(slices_num):
    slice_kspace = total_kspace[i]
    slice_kspace_tensor = T.to_tensor(slice_kspace)          # convert numpy to tensor
    slice_image = fastmri.ifft2c(slice_kspace_tensor)        # inverse fast FT
    slice_image_abs = fastmri.complex_abs(slice_image)        # compute the absolute value to get a real image
    image_list.append(slice_image_abs)
    slice_kspace_tensor_list.append(slice_kspace_tensor)    # 35* torch[640, 368])
    
image_list_tensor = torch.stack(image_list, dim=0)  # convert a list of tensor_image of a h5 file into tensor
print(image_list_tensor.shape)

sampled_image_list= []
for i in range(slices_num):
    slice_kspace_tensor = slice_kspace_tensor_list[i]
    masked_kspace, mask = T.apply_mask(slice_kspace_tensor, mask_func)
    Ny, Nx, _ = slice_kspace_tensor.shape
    mask = mask.repeat(Ny, 1, 1).squeeze()
    # functions.show_slice(mask, cmap='gray')
    # functions.show_slice(image_list[10], cmap='gray')
    sampled_image = fastmri.ifft2c(masked_kspace)          # inverse fast FT to get the complex image
    sampled_image_abs = fastmri.complex_abs(sampled_image)
    sampled_image_list.append(sampled_image_abs)
sampled_image_list_tensor = torch.stack(sampled_image_list, dim=0)
print(sampled_image_list_tensor.shape)


'''


def load_data(file_dir_path):
    file_path = get_files(file_dir_path)
    file_num = len(file_path)
    total_image_list = []
    total_sampled_image_list = []
    for h5_num in range(file_num):
        total_kspace, slices_num = load_dataset(file_path[0])
        image_list = []
        slice_kspace_tensor_list = []
        for i in range(slices_num):
            slice_kspace = total_kspace[i]
            slice_kspace_tensor = T.to_tensor(slice_kspace)  # convert numpy to tensor
            slice_image = fastmri.ifft2c(slice_kspace_tensor)  # inverse fast FT
            slice_image_abs = fastmri.complex_abs(slice_image)  # compute the absolute value to get a real image
            image_list.append(slice_image_abs)
            slice_kspace_tensor_list.append(slice_kspace_tensor)  # 35* torch[640, 368])

        image_list_tensor = torch.stack(image_list, dim=0)    # torch.Size([35, 640, 368])
        total_image_list.append(image_list_tensor)
        mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4])  # create the mask function object
        sampled_image_list = []
        for i in range(slices_num):
            slice_kspace_tensor = slice_kspace_tensor_list[i]
            masked_kspace, mask = T.apply_mask(slice_kspace_tensor, mask_func)
            Ny, Nx, _ = slice_kspace_tensor.shape
            mask = mask.repeat(Ny, 1, 1).squeeze()
            # functions.show_slice(mask, cmap='gray')
            # functions.show_slice(image_list[10], cmap='gray')
            sampled_image = fastmri.ifft2c(masked_kspace)  # inverse fast FT to get the complex image
            sampled_image_abs = fastmri.complex_abs(sampled_image)
            sampled_image_list.append(sampled_image_abs)
        sampled_image_list_tensor = torch.stack(sampled_image_list, dim=0)          # torch.Size([35, 640, 368])
        total_sampled_image_list.append(sampled_image_list_tensor)
    # total_image_tensor = torch.cat(total_image_list, dim=0)                       # torch.Size([6965, 640, 368])
    # total_sampled_image_tensor = torch.cat(total_sampled_image_list, dim=0)       # torch.Size([6965, 640, 368])
    total_image_tensor = torch.stack(total_image_list, dim=0)                       # torch.Size([199, 35, 640, 368])
    total_sampled_image_tensor = torch.stack(total_sampled_image_list, dim=0)       # torch.Size([199, 35, 640, 368])
    print(total_image_tensor.shape)
    print(total_sampled_image_tensor.shape)
    return total_image_tensor, total_sampled_image_tensor


a, b = load_data(file_dir_path=file_dir)

