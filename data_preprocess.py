#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
join = os.path.join
import argparse

from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import math

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
   
    interior_temp = np.zeros_like(inst_map,dtype=np.uint8)
    for i in range(1,inst_map.max()+1):
        mask = (inst_map==i)
        t = math.ceil(np.sqrt(np.sum(mask))/40)
    
        boundary = segmentation.find_boundaries(mask, mode='inner')
        # 细胞边界宽度与细胞大小成正比
        boundary = morphology.binary_dilation(boundary, morphology.disk(t))
        interior_temp[boundary]=2
        mask = np.logical_and(~boundary, mask)
        mask = morphology.remove_small_objects(mask, min_size=16)
        interior_temp[mask] = 1
    return interior_temp

   

def main():
    parser = argparse.ArgumentParser('Preprocessing for microscopy image segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='/data1/dataset/NeurIPS_CellSegData/Train_Labeled', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='/data1/dataset/NeurIPS_CellSegData/Train_Pre_3class_v2.1', type=str, help='preprocessing data path')    
    args = parser.parse_args()
    
    source_path = args.input_path
    target_path = args.output_path
    
    img_path = join(source_path, 'images')
    gt_path =  join(source_path, 'labels')
    
    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.split('.')[0]+'_label.tiff' for img_name in img_names]
    
    pre_img_path = join(target_path, 'images')
    pre_gt_path = join(target_path, 'labels')
    os.makedirs(pre_img_path, exist_ok=True)
    os.makedirs(pre_gt_path, exist_ok=True)
    
    for img_name, gt_name in zip(tqdm(img_names), gt_names):
        if img_name.endswith('.tif') or img_name.endswith('.tiff'):
            img_data = tif.imread(join(img_path, img_name))
        else:
            img_data = io.imread(join(img_path, img_name))
        gt_data = tif.imread(join(gt_path, gt_name))
        
        # normalize image data
        if len(img_data.shape) == 2:
            img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
        elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
            img_data = img_data[:,:, :3]
        else:
            pass
        pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
        for i in range(3):
            img_channel_i = img_data[:,:,i]
            if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
        
        # conver instance bask to three-class mask: interior, boundary
        interior_map = create_interior_map(gt_data.astype(np.int16))
        
        io.imsave(join(target_path, 'images', img_name.split('.')[0]+'.png'), pre_img_data.astype(np.uint8), check_contrast=False)
        io.imsave(join(target_path, 'labels', gt_name.split('.')[0]+'.png'), interior_map.astype(np.uint8), check_contrast=False)
    
if __name__ == "__main__":
    main()























