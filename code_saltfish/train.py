#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""
from pickle import FALSE
import sys
import argparse
import os

join = os.path.join

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import decollate_batch, PILReader,NumpyReader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    EnsureTyped,
    EnsureType,
    ResizeWithPadOrCropd,
)
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

from torch.autograd import Variable
from net import ResUNETR_s2,ResUNETR_s2widetiny

print("Successfully imported all requirements!")


class wL1Loss(nn.Module):
    def __init__(self):
        super(wL1Loss,self).__init__()
        
    def forward(self,inputs,weight):
        weight2 = (weight==0)
        weight1 = (weight>0)
        num1 = torch.sum(weight2 )
        num2 = torch.sum(weight1 )
        loss = torch.sum(torch.abs(inputs*weight2))/(num1+1e-3)+torch.sum(torch.abs(inputs*weight1-weight))/(num2+1e-3)
        return loss


def main():
    parser = argparse.ArgumentParser("Baseline for Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/data1/dataset/NeurIPS_CellSegData/Train_Pre_3class_v2.1",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--weight_path",
        default="/data1/dataset/NeurIPS_CellSegData/mesmer2/",
        type=str,
        help="training distance transform weight path",
    )
    parser.add_argument(
        "--work_dir", default="./work_dir", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=4, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name", default="res50_unetr", help="select mode: res50_unetr, res50wt_unetr,"
    )
    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=512, type=int, help="segmentation classes"
    )
    # Training parameters
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--val_interval", default=3, type=int)
    parser.add_argument("--epoch_tolerance", default=100, type=int)
    parser.add_argument("--initial_lr", type=float, default=6e-4, help="learning rate")

    
    parser.add_argument("--pre_trained_model_path", type=str, default=None, help="pretrain_model")
    
    args = parser.parse_args()

    monai.config.print_config()

    #%% set training/validation split
    np.random.seed(args.seed)
    model_path = join(args.work_dir, args.model_name)
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )
    img_path = join(args.data_path, "images")
    gt_path = join(args.data_path, "labels")

    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.split(".")[0] + "_label.png" for img_name in img_names]
    img_num = len(img_names)
    val_frac = 0.0
    indices = np.arange(img_num)
    np.random.shuffle(indices)
    val_split = int(img_num * val_frac)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

   
    weight_path = args.weight_path 
    direct_names = [gt_name.split(".")[0] + ".npy" for gt_name in gt_names]
    train_files = [
        {"img": join(img_path, img_names[i]), "label": join(gt_path, gt_names[i] ), 
        "weight": join(weight_path,direct_names[i])}
        for i in train_indices
    ]
    val_files = [
        {"img": join(img_path, img_names[i]), "label": join(gt_path, gt_names[i]), 
       "weight": join(weight_path,direct_names[i])}
        for i in val_indices
    ]
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )
    #%% define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["img", "label"], reader=PILReader, dtype=np.uint8
            ),  # image three channels (H, W, 3); label: (H, W)
            LoadImaged(
                keys=["weight"],reader=NumpyReader, dtype=np.float32
            ),
            AddChanneld(keys=["label","weight"], allow_missing_keys=True),  # label: (1, H, W)
            AsChannelFirstd(
                keys=["img"], channel_dim=-1, allow_missing_keys=True
            ),  # image: (3, H, W)
            ScaleIntensityd(
                keys=["img"], allow_missing_keys=True
            ),  # Do not scale label
            RandZoomd(
                keys=["img", "label","weight"],
                prob=1,
                min_zoom=0.25,
                max_zoom=2,
                mode=["area", "nearest", "bilinear"],
                keep_size=False,
            ),
            SpatialPadd(keys=["img", "label","weight"], spatial_size=args.input_size,mode="reflect"),
            RandSpatialCropd(
                keys=["img", "label","weight"], roi_size=(args.input_size,args.input_size), random_size=False
            ),
           
            RandAxisFlipd(keys=["img", "label","weight"], prob=0.5),
            RandRotate90d(keys=["img", "label","weight"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
      
            EnsureTyped(keys=["img", "label","weight"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
            LoadImaged(
                keys=["weight"], reader=NumpyReader, dtype=np.float32
            ),
            AddChanneld(keys=["label","weight"], allow_missing_keys=True),
            AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img", "label","weight"]),
        ]
    )

    #% define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4)
    check_data = monai.utils.misc.first(check_loader)
    print(
        "sanity check:",
        check_data["img"].shape,
        torch.max(check_data["img"]),
        check_data["label"].shape,
        torch.max(check_data["label"]),
        check_data["weight"].shape,
        torch.min(check_data["weight"]),
    )

    #%% create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

 
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    if args.model_name.lower() == "res50_unetr":
        model = ResUNETR_s2(
                img_size=(args.input_size, args.input_size),
                in_channels=3,
                out_channels=args.num_class+1,
                feature_size=64,  # should be divisible by 12
                spatial_dims=2,
            ).to(device)
    if args.model_name.lower() == "res50wt_unetr":
        model = ResUNETR_s2widetiny(
                img_size=(args.input_size, args.input_size),
                in_channels=3,
                out_channels=args.num_class+1,
                feature_size=64,  # should be divisible by 12
                spatial_dims=2,
            ).to(device) 
    if args.pre_trained_model_path != None:
        checkpoint = torch.load(join(args.pre_trained_model_path, 'best_Dice_model.pth'), map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        
    loss_function = monai.losses.DiceFocalLoss(softmax = False)
    loss_function2 = wL1Loss()
    initial_lr = args.initial_lr
    optimizer = torch.optim.AdamW(model.parameters(), initial_lr)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95, last_epoch=-1)
   
    # start a typical PyTorch training
    max_epochs = args.max_epochs


    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(model_path)
    for epoch in range(1, max_epochs):
        model.train()
        epoch_loss = 0
        for step, batch_data in enumerate(train_loader, 1):
            inputs, labels,weight = batch_data["img"].to(device), batch_data["label"].to(device), batch_data["weight"].to(device)
            optimizer.zero_grad()
            outputs,dis = model(inputs)
           
            outputs =torch.softmax(outputs,dim=1)
            outputs = outputs[:,0:3,:,:]
            labels_onehot = monai.networks.one_hot(
                labels, args.num_class
            )  # (b,cls,256,256)
            loss1 = loss_function(outputs, labels_onehot) 
            loss2 = loss_function2(dis,weight)
            loss =loss1+loss2   
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss1.item(), epoch_len * epoch + step)
            writer.add_scalar("dis_loss", loss2.item(), epoch_len * epoch + step)
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        scheduler.step()
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        print("lr:",optimizer.param_groups[0]['lr'])  
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss_values,
        }
        if epoch%10==0:
            torch.save(checkpoint, join(model_path, "best_Dice_model.pth"))

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()
    torch.save(checkpoint, join(model_path, "final_model.pth"))
    np.savez_compressed(
        join(model_path, "train_log.npz"),
        val_dice=metric_values,
        epoch_loss=epoch_loss_values,
    )


if __name__ == "__main__":
    main()
