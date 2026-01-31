'''
For batch 32 with Accelerate

This file can replace the "train.amp"
'''

from torch_ema import ExponentialMovingAverage
from models.model_2d import ResNet18
from models.model_3d import I3D
from models.auxiliary import *
from dataloader_mask import get_data_loader
import logging
import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import random
import pandas
from experiment_config import config
from datetime import datetime
import os
from tqdm import tqdm
from copy import deepcopy
from preprocessing.utils import rot_flip_yz_2

from accelerate import Accelerator
from accelerate.utils import set_seed


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

def make_weights_for_balanced_classes(labels):
    # Making sampling weights for the data samples
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    weights = []
    for label in labels:
        weights.append(n_samples / float(cnt_dict[label]))
    return weights

class DiceLoss(nn.Module):
    # Dice Loss for segmentation tasks
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

def train_multitask(train_csv_path, valid_csv_path, exp_save_root):
    accelerator = Accelerator(mixed_precision='fp16')

    set_seed(config.SEED)

    if accelerator.is_main_process:
        for key, value in vars(config).items():
            logging.info(f"{key} : {value}")

    # Load original datasets
    train_df = pandas.read_csv(train_csv_path)
    valid_df = pandas.read_csv(valid_csv_path)
    
    # Combine all data first
    all_df = pandas.concat([train_df, valid_df], ignore_index=True)
    unique_patients = valid_df['PatientID'].unique()
    np.random.shuffle(unique_patients)
    
    cumulative_samples = 0
    selected_patients = []
    valid_samples_num = config.VALID_SAMPLES_NUM
    
    for patient in unique_patients:
        patient_samples = len(valid_df[valid_df['PatientID'] == patient])
        if cumulative_samples + patient_samples <= valid_samples_num:
            selected_patients.append(patient)
            cumulative_samples += patient_samples
        else:
            if abs(valid_samples_num - cumulative_samples) > abs(valid_samples_num - (cumulative_samples + patient_samples)):
                selected_patients.append(patient)
                cumulative_samples += patient_samples
            break
            
    if cumulative_samples < valid_samples_num and len(selected_patients) < len(unique_patients):
        remaining_patients = [p for p in unique_patients if p not in selected_patients]
        if remaining_patients:
            selected_patients.append(remaining_patients[0])
            
    new_valid_df = valid_df[valid_df['PatientID'].isin(selected_patients)].reset_index(drop=True)
    new_train_df = all_df[~all_df['PatientID'].isin(selected_patients)].reset_index(drop=True)
    
    logging.info(f"Original train samples: {len(train_df)}")
    
    if config.SUBMISSION == False and accelerator.is_main_process:
        logging.info(f"Original valid samples: {len(valid_df)}")
        logging.info(f"Total unique patients in validation set: {len(unique_patients)}")
        logging.info(f"Selected patients for validation: {len(selected_patients)}")
        logging.info(f"New train samples: {len(new_train_df)}")
        logging.info(f"New valid samples: {len(new_valid_df)} (target was {valid_samples_num})")
        
        logging.info(f"Number of malignant training samples: {new_train_df.label.sum()}")
        logging.info(f"Number of benign training samples: {len(new_train_df) - new_train_df.label.sum()}")
        logging.info(f"Number of malignant validation samples: {new_valid_df.label.sum()}")
        logging.info(f"Number of benign validation samples: {len(new_valid_df) - new_valid_df.label.sum()}")
    
    if config.SUBMISSION == False:
        weights = make_weights_for_balanced_classes(new_train_df.label.values)
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(new_train_df))
        
        train_loader = get_data_loader(
            config.DATADIR, config.MASK_DATADIR, new_train_df, 
            mode=config.MODE, sampler=sampler, workers=config.NUM_WORKERS, 
            batch_size=config.BATCH_SIZE, rotations=config.ROTATION, 
            translations=config.TRANSLATION, size_mm=config.SIZE_MM, size_px=config.SIZE_PX,
        )
    else:
        weights = make_weights_for_balanced_classes(train_df.label.values)
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_df))
        
        train_loader = get_data_loader(
            config.DATADIR, config.MASK_DATADIR, train_df,
            mode=config.MODE, sampler=sampler, workers=config.NUM_WORKERS,
            batch_size=config.BATCH_SIZE, rotations=config.ROTATION,
            translations=config.TRANSLATION, size_mm=config.SIZE_MM, size_px=config.SIZE_PX,
        )

    valid_loader = get_data_loader(
        config.DATADIR, config.MASK_DATADIR, new_valid_df,
        mode=config.MODE, workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE,
        rotations=None, translations=None, size_mm=config.SIZE_MM, size_px=config.SIZE_PX,
    )

    # device = torch.device(config.DEVICE)
    device = accelerator.device 

    # Build multi-task model
    if config.MODE == "2D":
        feature_extractor = ResNet18() 
    elif config.MODE == "3D":
        feature_extractor = I3D(
            num_classes=1, input_channels=3, pre_trained=True, 
            dropout_prob=config.DROP_RATE, freeze_bn=True, extract_feature=True, 
        ) 

    aux_model = SegmentationHead(in_channels=1024, out_channels=1, target_size=(64,64,64))
    model = MultiTaskModel(feature_extractor, aux_model=aux_model, aux_task=config.AUX_TASK)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.EMA_RATE)
        
    loss_function = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    if config.SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)
    elif config.SCHEDULER == "step_custom":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.DOWNSTEPS, gamma=0.3
        )

    # Wrap all objects with accelerate's prepare method
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
    
    # move EMA to accelerator device
    ema.to(device)

    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    epochs = config.EPOCHS
    patience = config.PATIENCE
    counter = 0

    for epoch in range(epochs):
        # logging은 main process에서만 수행
        if accelerator.is_main_process:
            logging.info("-" * 10)
            logging.info("epoch {}/{}".format(epoch + 1, epochs))

        # Train
        model.train()
        epoch_cls_loss = 0
        epoch_seg_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, masks, labels = batch_data["image"], batch_data["mask"], batch_data["label"]
            
            labels = labels.float() 
            
            benigns = torch.where(labels.squeeze() == 0)
            malignants = torch.where(labels.squeeze() == 1)
            
            if random.random() < config.RF_RATIO[0]:
                inputs[benigns], masks[benigns] = rot_flip_yz_2(inputs[benigns], masks[benigns])
            if random.random() < config.RF_RATIO[1]:
                inputs[malignants], masks[malignants] = rot_flip_yz_2(inputs[malignants], masks[malignants])

            inputs = torch.nn.functional.interpolate(inputs, size=(64, int(64*config.UP_SCALE), int(64*config.UP_SCALE)))
            

            optimizer.zero_grad()
            features = model.extract_feature(inputs)
            cls_outputs, seg_outputs = model(features_main=features)

            loss_cls = loss_function(cls_outputs.squeeze(), labels.squeeze())
            loss_seg = dice_loss(seg_outputs, masks)
            loss = loss_cls + config.AUX_LOSS_WEIGHT * loss_seg
            
            accelerator.backward(loss)
            optimizer.step()
            
            # EMA Update
            ema.update()
            
            epoch_cls_loss += loss_cls.item()
            epoch_seg_loss += loss_seg.item()
            
            if step % 100 == 0 and accelerator.is_main_process:
                epoch_len = len(new_train_df) // train_loader.batch_size
                logging.info(
                    "{}/{}, train_cls_loss: {:.4f}, train_seg_loss: {:.4f}".format(step, epoch_len, loss_cls.item(), loss_seg.item())
                )

        epoch_cls_loss /= step
        epoch_seg_loss /= step

        if accelerator.is_main_process:
            logging.info(
                "epoch {} average train cls loss: {:.4f}, train seg loss: {:.4f}".format(epoch + 1, epoch_cls_loss, epoch_seg_loss)
            )
        
        with ema.average_parameters():
            
            if config.SUBMISSION == True:
                if (epoch + 1) % 5 == 0:
                    accelerator.wait_for_everyone() 
                    
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(
                            unwrapped_model.state_dict(),
                            exp_save_root / f"multitask_model_epoch_{epoch + 1}.pth",
                        )
                        logging.info(f"Saved model at epoch {epoch + 1} for submission.")
            else:
                # Validate
                model.eval()
                epoch_cls_loss = 0
                epoch_seg_loss = 0
                step = 0

                val_preds_list = []
                val_targets_list = []

                with torch.no_grad():
                    for val_data in valid_loader:
                        step += 1
                        val_images, val_masks, val_labels = (
                            val_data["image"], 
                            val_data["mask"], 
                            val_data["label"],
                        )
                        val_labels = val_labels.float()
                        
                        val_images = torch.nn.functional.interpolate(val_images, size=(64, int(64*config.UP_SCALE), int(64*config.UP_SCALE)))
                        
                        features = model.extract_feature(val_images)
                        cls_outputs, seg_outputs = model(features_main=features)
                        
                        loss_cls = loss_function(cls_outputs.squeeze(), val_labels.squeeze())
                        loss_seg = dice_loss(seg_outputs, val_masks)
                        
                        epoch_cls_loss += loss_cls.item()
                        epoch_seg_loss += loss_seg.item()
                        
                        gathered_preds, gathered_labels = accelerator.gather_for_metrics((cls_outputs, val_labels))
                        
                        val_preds_list.append(gathered_preds.detach().cpu())
                        val_targets_list.append(gathered_labels.detach().cpu())

                    epoch_cls_loss /= step
                    epoch_seg_loss /= step
                    
                    if accelerator.is_main_process:
                        logging.info(
                            "epoch {} average valid cls loss: {:.4f}, valid seg loss: {:.4f}".format(epoch + 1, epoch_cls_loss, epoch_seg_loss)
                        )

                    y_pred_all = torch.cat(val_preds_list)
                    y_all = torch.cat(val_targets_list)
                    
                    y_pred_np = torch.sigmoid(y_pred_all.reshape(-1)).numpy().reshape(-1)
                    y_np = y_all.numpy().reshape(-1)

                    if accelerator.is_main_process:
                        fpr, tpr, _ = metrics.roc_curve(y_np, y_pred_np)
                        auc_metric = metrics.auc(fpr, tpr)

                        if auc_metric > best_metric:
                            counter = 0
                            best_metric = auc_metric
                            best_metric_epoch = epoch + 1

                            unwrapped_model = accelerator.unwrap_model(model)
                            torch.save(
                                unwrapped_model.state_dict(),
                                exp_save_root / "best_multitask_model.pth",
                            )
                            logging.info("saved new best multitask model")

                        logging.info(
                            "current epoch: {} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                                epoch + 1, auc_metric, best_metric, best_metric_epoch
                            )
                        )
        
        counter += 1
        scheduler.step()

    if accelerator.is_main_process:
        logging.info(
            "Multi-task training completed, best_metric: {:.4f} at epoch: {}".format(
                best_metric, best_metric_epoch
            )
        )
    
    return exp_save_root / "best_multitask_model.pth"

def main():
    experiment_name = f"{config.EXPERIMENT_NAME}-multitask-{config.MODE}-{datetime.today().strftime('%Y%m%d')}"
    
    exp_save_root = config.EXPERIMENT_DIR / experiment_name
    exp_save_root.mkdir(parents=True, exist_ok=True)
    
    train_multitask(
        config.CSV_DIR_TRAIN,
        config.CSV_DIR_VALID,
        exp_save_root
    )
    
    logging.info(f"All results saved in: {exp_save_root}")

if __name__ == "__main__":
    main()