from pickletools import uint8
import sys
from typing import Tuple

import os
join = os.path.join
import argparse
import numpy as np
import torch
import monai
from monai.inferers import sliding_window_inference
import time
from skimage import io, segmentation, morphology, measure, exposure
from skimage.feature import peak_local_max
import tifffile as tif
import skimage
import cv2
from net import ResUNETR_s2,ResUNETR_s2widetiny
import  torch.nn.functional  as F

def output(imb1,imb,dis):
    imb1 = morphology.remove_small_objects(imb1,16)

    d =  int(np.sum(imb1)/np.max(measure.label(imb1)))
    d1= int(np.sqrt(d))
    d1+=d1%2+1
   
    im4 = cv2.GaussianBlur(dis,(d1,d1),0)
    im5 = peak_local_max(im4,indices=False,min_distance =int(d1/4),threshold_rel=0.6)
    imc = cv2.applyColorMap((imb).astype(np.uint8),1)
    #imx= measure.label(((im4>0.8))&imb1)
    imx= measure.label(((im4>0.85*np.max(im4)))&imb1)
    im6 =  cv2.watershed(cv2.applyColorMap((imb1).astype(np.uint8),1),imx+(imx!=0).astype(np.int32)+1-imb)-1
    imb1[im6!=0]=0
    im6[im6<0]=0
    imx = measure.label(im5 &imb1)
    im7 =  cv2.watershed(cv2.applyColorMap((imb1).astype(np.uint8),1),imx+(imx!=0).astype(np.int32)+1-imb)-1
    im7[im7<0]=0
    imx = im6+im7+np.max(im6)*(im7!=0)
    im6 =  cv2.watershed(imc,imx+(imx!=0).astype(np.int32)+1-imb)-1

    im6[im6<0]=0
    im7 = im6>0
    d = int(np.sum(im7)/np.max(im6))
    im7 = morphology.remove_small_objects(im7,int(d/10))
    return im6*im7

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='/data1/dataset/NeurIPS_CellSegData/TuningSet', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
   

    # Model parameters
    parser.add_argument('--num_class', default=3, type=int, help='segmentation classes')
    parser.add_argument('--input_size', default=512, type=int, help='segmentation classes')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    img_names = sorted(os.listdir(join(input_path)))
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    model = ResUNETR_s2widetiny(
            img_size=(args.input_size, args.input_size),
            in_channels=3,
            out_channels=args.num_class+1,
            feature_size=64,  # should be divisible by 12
            spatial_dims=2,
        ).to(device)
    checkpoint = torch.load(join('./work_dir/res50wt_unetr', 'best_Dice_model.pth'), map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model2 = ResUNETR_s2(
            img_size=(args.input_size, args.input_size),
            in_channels=3,
            out_channels=args.num_class+1,
            feature_size=64,  # should be divisible by 12
            spatial_dims=2,
        ).to(device)
    checkpoint = torch.load(join('./work_dir/res50_unetr', 'best_Dice_model.pth'), map_location=torch.device(device))
    model2.load_state_dict(checkpoint['model_state_dict'])
    #%%
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 4
    model.eval()
    model2.eval()
    with torch.no_grad():
        for img_name in img_names:
            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(input_path, img_name))
            else:
                img_data = io.imread(join(input_path, img_name))

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
                    
            t0 = time.time()
            test_npy01 = pre_img_data/np.max(pre_img_data)
           
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)

            b,c,h,w = test_tensor.size()
            dis =torch.zeros(b,1,h,w).to(device)
            test_pred_out=torch.zeros(b,4,h,w).to(device)
            if h*w <25000000:   
                for scale in [1,1.25,1.5]:
                    test_tensor1 =F.interpolate(test_tensor, (int(scale*h),int(scale*w)), mode='bilinear', align_corners=True)
                    test_pred_out1 ,dis1= sliding_window_inference(test_tensor1, roi_size, sw_batch_size, model, padding_mode= "reflect")
                    test_pred_out1 = F.interpolate(test_pred_out1, (h,w), mode='bilinear', align_corners=True)
                    dis1 = F.interpolate(dis1, (h,w), mode='bilinear', align_corners=True)
                    test_pred_out2 ,dis2= sliding_window_inference(test_tensor1,roi_size, sw_batch_size, model2, padding_mode= "reflect")
                    test_pred_out2 = F.interpolate(test_pred_out2, (h,w), mode='bilinear', align_corners=True)
                    dis2 = F.interpolate(dis2, (h,w), mode='bilinear', align_corners=True)
                    dis+=(dis1+dis2)
                    test_pred_out+=(test_pred_out1+test_pred_out2)
                dis/=6
                test_pred_out/=6
                    
            else:

                test_pred_out1 ,dis1= sliding_window_inference(test_tensor, roi_size, sw_batch_size, model, padding_mode= "reflect")
                test_pred_out1 = F.interpolate(test_pred_out1, (h,w), mode='bilinear', align_corners=True)
                dis1 = F.interpolate(dis1, (h,w), mode='bilinear', align_corners=True)
                test_pred_out2 ,dis2= sliding_window_inference(test_tensor,roi_size, sw_batch_size, model2, padding_mode= "reflect")
                test_pred_out2 = F.interpolate(test_pred_out2, (h,w), mode='bilinear', align_corners=True)
                dis2 = F.interpolate(dis2, (h,w), mode='bilinear', align_corners=True)
                dis+=(dis1+dis2)
                test_pred_out+=(test_pred_out1+test_pred_out2)
                dis/=2
                test_pred_out/=2
          
            test_pred_out= torch.nn.functional.softmax(test_pred_out, dim=1)
       
            dis = dis[0,0].cpu().numpy()
          
            p0 = test_pred_out[0,0].cpu().numpy()
            p1 = test_pred_out[0,1].cpu().numpy()
  
            test_pred_npy =(p0<0.3)
            test_pred_npy1 = (p1>0.7)
       
            out = output(test_pred_npy1 ,test_pred_npy,dis)

            tif.imwrite(join(output_path, img_name.split('.')[0]+'_label.tiff'),out, compression='zlib')
           
            t1 = time.time()
            print(f'Prediction finished: {img_name}; img size = {pre_img_data.shape}; costing: {t1-t0:.2f}s')
            
         
        
if __name__ == "__main__":
    main()





