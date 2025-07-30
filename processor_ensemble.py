# For ensemble
import numpy as np
import torch
import torch.nn as nn
import os
import logging
from experiment_config import config

import dataloader
from models.model_3d import I3D
from models.model_2d import ResNet18
from models.auxiliary import MultiTaskModel, SegmentationHead

logging.basicConfig(
    level=logging.INFO, 
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

class MalignancyProcessor:

    def __init__(self, mode="3D", suppress_logs=False, model_name="LUNA25-baseline-2D"):
        self.size_px = 64
        self.size_mm = 50
        self.mode = mode
        self.suppress_logs = suppress_logs

        self.model_root = "/opt/app/resources/"
        self.model_paths = []

        pths = ['multitask_model_epoch_15.pth', 
                'multitask_model_epoch_20.pth', 
                'multitask_model_epoch_25.pth', 
                'multitask_model_epoch_30.pth', ]
        
        for pth in pths:
            self.model_paths.append(
                os.path.join(
                self.model_root,
                model_name,
                pth,
            ))
        
        if not self.suppress_logs:
            logging.info(f"Initializing deep learning system for ensembling {len(self.model_paths)} models.")

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):

        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            coord_space_world=True,
            mode=mode,
        )

        patch = patch.astype(np.float32)

        patch = dataloader.clip_and_scale(patch)
        return patch

    def _process_model(self, mode):
        if not self.suppress_logs:
            logging.info(f"Processing in {mode} mode with ensembling")

        if mode == "2D":
            output_shape = [1, self.size_px, self.size_px]
        else: 
            output_shape = [self.size_px, self.size_px, self.size_px]

        nodules = np.array([self.extract_patch(c, output_shape, mode) for c in self.coords])
        nodules = torch.from_numpy(nodules).cuda()
        
        nodules = torch.nn.functional.interpolate(nodules, size=(64, int(64*config.UP_SCALE), int(64*config.UP_SCALE)))

        all_logits = [] 

        for model_path in self.model_paths:
            if not self.suppress_logs:
                logging.info(f"  -> Running inference with model: {model_path}")
            
            if self.mode == "2D":
                model = ResNet18().cuda()
            else: # 3D
                feature_extractor = I3D(num_classes=1, input_channels=3, pre_trained=False, extract_feature=True).cuda()
                aux_model = SegmentationHead(in_channels=1024, out_channels=1, target_size=(64,64,64)).cuda()
                model = MultiTaskModel(feature_extractor, aux_model=aux_model).cuda()
            
            ckpt = torch.load(model_path, map_location="cuda:0")
            
            model.load_state_dict(ckpt, strict=False)
            model.eval()

            with torch.no_grad():
                features = model.extract_feature(nodules)
                logits, _ = model(features, validate=True)
                all_logits.append(logits.cpu().numpy())
        
        if not self.suppress_logs:
            logging.info(f"Ensembling results by averaging logits.")
            
        ensembled_logits = np.mean(all_logits, axis=0)
        
        return ensembled_logits

    def predict(self):
        logits = self._process_model(self.mode)
        probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return probability, logits