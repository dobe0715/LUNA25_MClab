import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




def get_padding_shape(filter_shape, stride): 
    def _pad_top_bottom(filter_dim, stride_val): 
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes): 
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init 


class Unit3Dpy(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        activation="relu",
        padding="SAME",
        use_bias=False,
        use_bn=True,
    ):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == "SAME":
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == "VALID":
            padding_shape = 0
        else:
            raise ValueError(
                "padding should be in [VALID|SAME] but got {}".format(padding)
            )

        if padding == "SAME":
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size, stride=stride, bias=use_bias
                )
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias,
                )
        elif padding == "VALID":
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias,
            )
        else:
            raise ValueError(
                "padding should be in [VALID|SAME] but got {}".format(padding)
            )

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == "relu":
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == "SAME" and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out



class SegmentationHead(nn.Module):
    """특징 맵을 입력받아 분할 마스크를 예측하는 분할 헤드"""
    def __init__(self, in_channels=1024, out_channels=1, target_size=(64, 64, 64)):
        super().__init__()
        self.target_size = target_size
        
        self.block1 = Unit3Dpy(
            in_channels=in_channels,
            out_channels=832,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="SAME",
            use_bias=False,
            use_bn=True
        )
        
        self.block2 = Unit3Dpy(
            in_channels=832,
            out_channels=480,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="SAME",
            use_bias=False,
            use_bn=True
        )
        
        self.block3 = Unit3Dpy(
            in_channels=480,
            out_channels=192,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="SAME",
            use_bias=False,
            use_bn=True
        )
        
        self.block4 = Unit3Dpy(
            in_channels=192,
            out_channels=64,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="SAME",
            use_bias=False,
            use_bn=True
        )
        
        self.block5 = Unit3Dpy(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="SAME",
            use_bias=False,
            use_bn=True
        )

        self.last_layer = nn.Conv3d(64, out_channels, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
        
    def forward(self, features):
        out = self.block1(features[0])
        out = F.interpolate(out, size=(16, 8, 8), mode='trilinear', align_corners=False) 
        
        out = self.block2(out + features[1])
        out = F.interpolate(out, size=(32, 16, 16), mode='trilinear', align_corners=False)
        
        out = self.block3(out + features[2])
        out = F.interpolate(out, size=(32, 32, 32), mode='trilinear', align_corners=False)
        
        out = self.block4(out + features[3])
        out = F.interpolate(out, size=(32, 64, 64), mode='trilinear', align_corners=False)
        
        out = self.block5(out + features[4])
        out = F.interpolate(out, size=self.target_size, mode='trilinear', align_corners=False)
        
        out = self.last_layer(out)
        
        return out


class MultiTaskModel(nn.Module):
    def __init__(self, feature_extractor, aux_model, aux_task):
        super().__init__()
        self.aux_task = aux_task
        self.backbone = feature_extractor

        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)), 
            nn.Flatten(), 
            nn.Linear(1024, 1),
        )

        
        self.aux_model = aux_model
        
    def extract_feature(self, images):
        return self.backbone(images)

    def forward(self, features_main, validate=False):
        
        main_logits = self.classifier_head(features_main[0])
        
        aux_outputs = None
        if validate == False:
            aux_outputs = self.aux_model(features_main)
            
            
        return main_logits, aux_outputs