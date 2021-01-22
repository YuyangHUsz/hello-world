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
import random
from torch.utils.data import TensorDataset, DataLoader


def get_files(file_dir):
    file_path = []
    for fname in sorted(os.listdir(file_dir)):
        subject_path = os.path.join(file_dir, fname)
        file_path.append(subject_path)
   # print('there are ' + str(len(file_path)) + ' files')
    return file_path


def load_dataset(file_name):
    dataset = h5py.File(file_name, 'r')
    volume_kspace = dataset['kspace'][()]
    target = dataset['reconstruction_esc'][()]
    #print(target.shape)
    #print(volume_kspace.shape)
    slice_num = volume_kspace.shape[0]
    return volume_kspace, slice_num, target


def load_data_from_pathlist(path):
    file_num = len(path)
    use_num = file_num//3
    total_target_list = []
    total_sampled_image_list = []
    for h5_num in range(use_num):
        total_kspace, slices_num, target = load_dataset(path[h5_num])
        image_list = []
        slice_kspace_tensor_list = []
        target_image_list = []
        for i in range(slices_num):
            slice_kspace = total_kspace[i]
            #target_image = target[i]
            slice_kspace_tensor = T.to_tensor(slice_kspace)  # convert numpy to tensor
            slice_kspace_tensor = slice_kspace_tensor.float()
            #print(slice_kspace_tensor.shape)
            slice_kspace_tensor_list.append(slice_kspace_tensor)  # 35* torch[640, 368])
            #target = target_image_list.append(target_image)

        #image_list_tensor = torch.stack(image_list, dim=0)  # torch.Size([35, 640, 368])
        #total_image_list.append(image_list_tensor)
        mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4])  # create the mask function object
        sampled_image_list = []
        target_list = []
        for i in range(slices_num):
            slice_kspace_tensor = slice_kspace_tensor_list[i]
            masked_kspace, mask = T.apply_mask(slice_kspace_tensor, mask_func)
            Ny, Nx, _ = slice_kspace_tensor.shape
            mask = mask.repeat(Ny, 1, 1).squeeze()
            # functions.show_slice(mask, cmap='gray')
            # functions.show_slice(image_list[10], cmap='gray')
            sampled_image = fastmri.ifft2c(masked_kspace)  # inverse fast FT to get the complex image
            sampled_image = T.complex_center_crop(sampled_image, (320, 320))
            sampled_image_abs = fastmri.complex_abs(sampled_image)
            sampled_image_list.append(sampled_image_abs)
        sampled_image_list_tensor = torch.stack(sampled_image_list, dim=0)  # torch.Size([35, 640, 368])
        total_sampled_image_list.append(sampled_image_list_tensor)
        target = T.to_tensor(target)
        total_target_list.append(target)
    #target_image_tensor = torch.cat(target_image_list, dim=0)                       # torch.Size([6965, 640, 368])
    total_target = torch.cat(total_target_list, dim=0)
    total_sampled_image_tensor = torch.cat(total_sampled_image_list, dim=0)       # torch.Size([6965, 640, 368])
    total_sampled_image_tensor, mean, std = T.normalize_instance(total_sampled_image_tensor, eps=1e-11)
    total_sampled_image_tensor = total_sampled_image_tensor.clamp(-6, 6)
    target_image_tensor = T.normalize(total_target, mean, std, eps=1e-11)
    target_image_tensor = target_image_tensor.clamp(-6, 6)
    # total_image_tensor = torch.stack(total_image_list, dim=0)  # torch.Size([199, 35, 640, 368])
    # total_sampled_image_tensor = torch.stack(total_sampled_image_list, dim=0)  # torch.Size([199, 35, 640, 368])
    #print(target_image_tensor.shape)
    #print(total_sampled_image_tensor.shape)
    return target_image_tensor, total_sampled_image_tensor


#file_dir = '/Users/huyuyang/singlecoil_val'


def load_data_final(file_dir_path, train_scale=0.8, val_scale=0.1):
    file_path = get_files(file_dir_path)
    random.shuffle(file_path)
    file_num = len(file_path)
    train_file_num = int(file_num*train_scale)
    val_file_num = int(file_num * val_scale)
    test_file_num = file_num - train_file_num - val_file_num
    train_file_path = file_path[:train_file_num]
    val_file_path = file_path[train_file_num:train_file_num+test_file_num]
    test_file_path = file_path[train_file_num+test_file_num : ]
    training_data, train_gt = load_data_from_pathlist(train_file_path)
    validation_data, validation_gt = load_data_from_pathlist(val_file_path)
    testing_data, testing_gt = load_data_from_pathlist(test_file_path)
    return training_data,train_gt, validation_data,validation_gt, testing_data, testing_gt






'''''''''
print(training_data.shape)
print(train_gt.shape)
print(validation_data.shape)
print(validation_gt.shape)
print(testing_data.shape)
print(testing_gt.shape)
there are 199 files
torch.Size([5724, 640, 372])
torch.Size([5724, 640, 372])
torch.Size([756, 640, 368])
torch.Size([756, 640, 368])
torch.Size([779, 640, 368])
torch.Size([779, 640, 368])
torch.Size([5724, 640, 372])
torch.Size([5724, 640, 372])
torch.Size([756, 640, 368])
torch.Size([756, 640, 368])
torch.Size([779, 640, 368])
torch.Size([779, 640, 368])
'''