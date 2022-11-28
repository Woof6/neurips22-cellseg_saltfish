from re import T
from typing import Sequence, Tuple, Type, Union
import torch
import torch.nn as nn

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, optional_import
from torchvision import models

rearrange, _ = optional_import("einops", name="rearrange")

"""
resnet backbone
"""

class ResUNETR_s2(nn.Module):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        feature_size: int = 64,
        feature_size2: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        normalize: bool = True,
        spatial_dims: int = 3,
      
    ) -> None:
        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.normalize = normalize
      
        
        self.res = models.resnet50(pretrained=True)
        del self.res.fc
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
      
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32 * feature_size,
            out_channels=16* feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size2,
            out_channels=8 * feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder5_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size2,
            out_channels=8 * feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 8,
            out_channels=feature_size2 * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 8,
            out_channels=feature_size2 * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 4,
            out_channels=feature_size2 * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 4,
            out_channels=feature_size2 * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size2, out_channels=out_channels
        )  # type: ignore
        self.out_2 = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size2, out_channels=1
        )  # type: ignore

        
    def forward(self, x_in):
        x = self.res.conv1(x_in)
        x = self.res.bn1(x)
        x0 = self.res.relu(x)
        x = self.res.maxpool(x0)
        x1 = self.res.layer1(x)
        x2 = self.res.layer2(x1)
        x3 = self.res.layer3(x2)
        x4 = self.res.layer4(x3)
        hidden_states_out = [x0,x1,x2,x3,x4]
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])
        enc5 = self.encoder10(hidden_states_out[4])
        ### 分类
        dec4 = self.decoder5(enc5,enc4 )
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        dec0 = self.decoder1(dec1, enc0)
        logits = self.out(dec0)
     
        ### 回归
        dec4_2 = self.decoder5_2(enc5,enc4 )
        dec3_2 = self.decoder4_2(dec4_2, enc3)
        dec2_2 = self.decoder3_2(dec3_2, enc2)
        dec1_2 = self.decoder2_2(dec2_2, enc1)
        dec0_2 = self.decoder1_2(dec1_2, enc0)
        logits_2 = self.out_2(dec0_2)
      
        return logits,logits_2



class ResUNETR_s2widetiny(nn.Module):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        feature_size: int = 64,
        feature_size2: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        normalize: bool = True,
        spatial_dims: int = 3,
      
    ) -> None:
        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.normalize = normalize
      
     
        self.res = models.wide_resnet50_2(pretrained=True)
        del self.res.fc
        del self.res.layer4
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 8,
            out_channels=feature_size2 * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 8,
            out_channels=feature_size2 * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 4,
            out_channels=feature_size2 * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 4,
            out_channels=feature_size2 * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2 * 2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size2, out_channels=out_channels
        )  # type: ignore
        self.out_2 = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size2, out_channels=1
        )  # type: ignore

        
    def forward(self, x_in):
        x = self.res.conv1(x_in)
        x = self.res.bn1(x)
        x0 = self.res.relu(x)
        x = self.res.maxpool(x0)
        x1 = self.res.layer1(x)
        x2 = self.res.layer2(x1)
        x3 = self.res.layer3(x2)
        hidden_states_out = [x0,x1,x2,x3]
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])
        ### 分类
        dec3 = self.decoder4(enc4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        dec0 = self.decoder1(dec1, enc0)
        logits = self.out(dec0)
        ### 回归
        dec3_2 = self.decoder4_2(enc4, enc3)
        dec2_2 = self.decoder3_2(dec3_2, enc2)
        dec1_2 = self.decoder2_2(dec2_2, enc1)
        dec0_2 = self.decoder1_2(dec1_2, enc0)
        logits_2 = self.out_2(dec0_2)
      
        return logits,logits_2

class ResUNETR_final(nn.Module):
    def __init__(self, input_size, num_class):
        super(ResUNETR_final, self).__init__()
        self.branch1 = ResUNETR_s2(
            img_size=(input_size, input_size),
            in_channels=3,
            out_channels=num_class+1,
            feature_size=64,  
            spatial_dims=2,
        )
        self.branch2 = ResUNETR_s2widetiny(
            img_size=(input_size, input_size),
            in_channels=3,
            out_channels=num_class+1,
            feature_size=64,  
            spatial_dims=2,
        )

    def forward(self, data):
        l1,d1 = self.branch1(data)
        l2,d2 = self.branch2(data)

        return l1+l2,d1+d2