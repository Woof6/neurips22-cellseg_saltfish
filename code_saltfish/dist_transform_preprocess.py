import numpy as np
import os
from tqdm import tqdm
import os
join = os.path.join
import argparse

from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2 
import matplotlib.pyplot as plt
import sys
    
def main():
    parser = argparse.ArgumentParser('Dist transform for microscopy image segmentation', add_help=False)
    parser.add_argument(
        "--data_path",
        default="/data1/dataset/NeurIPS_CellSegData/Train_Pre_3class_v2.1",
        type=str,
        help="prerocessed training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--data_ori_path",
        default="/data1/dataset/NeurIPS_CellSegData/Train_Labeled",
        type=str,
        help="original training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--save_path",
        default='/data1/dataset/NeurIPS_CellSegData/mesmer2',
        type=str,
        help="dist transform save path",
    )
    args = parser.parse_args()
    img_path = join(args.data_path, "images")
    gt_path = join(args.data_path, "labels")
    gt2_path =  join(args.data_ori_path, "labels")

    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.split('.')[0]+'_label.png' for img_name in img_names]
    gt2_names = [img_name.split('.')[0]+'_label.tiff' for img_name in img_names]
    
    pre_weight = args.save_path
    os.makedirs(pre_weight, exist_ok=True)
    
    
    for  gt_name ,gt2_name in zip(tqdm(gt_names),gt2_names):
        gt_data = cv2.imread(join(gt_path, gt_name),cv2.IMREAD_GRAYSCALE)
        gt_data = 1-np.square(1-gt_data)
        dist_transform = cv2.distanceTransform(gt_data,cv2.DIST_L2,0)
        gt_data2 = tif.imread(join(gt2_path, gt2_name))
        weight_map =np.zeros_like(gt_data2)
        weight_map = weight_map.astype(float)
        for i in range(1,gt_data2.max()+1):
            mask = (gt_data2==i)
            weight_map+= mask*(1/(1+(np.max(mask*dist_transform)-mask*dist_transform)/np.sqrt(np.sum(mask))))
           
        np.save(join(pre_weight, gt_name.split('.')[0]), weight_map.astype(np.float32).swapaxes(0,1))
      
    
if __name__ == "__main__":
    main()
    

