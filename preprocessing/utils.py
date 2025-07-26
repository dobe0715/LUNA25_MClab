import pandas as pd
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def standardize_nodule_df(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset.PatientID = dataset.PatientID.astype(int)
    dataset.LesionID = dataset.LesionID.astype(int)
    dataset.StudyDate = dataset.StudyDate.astype(int)
    dataset["NoduleID"] = [
        f"{p}_{l}" for p, l in zip(dataset.PatientID, dataset.LesionID)
    ]
    dataset["AnnotationID"] = [
        f"{n}_{s}" for n, s in zip(dataset.NoduleID, dataset.StudyDate)
    ]
    return dataset

def transform(input_image, point):
    return np.array(
        list(
            reversed(
                input_image.TransformContinuousIndexToPhysicalPoint(
                    list(reversed(point))
                )
            )
        )
    )


def itk_image_to_numpy_image(input_image: sitk.Image) -> Tuple[np.array, Dict]:
    numpyImage = sitk.GetArrayFromImage(input_image)
    numpyOrigin = np.array(list(reversed(input_image.GetOrigin())))
    numpySpacing = np.array(list(reversed(input_image.GetSpacing())))

    # get numpyTransform
    tNumpyOrigin = transform(input_image, np.zeros((numpyImage.ndim,)))
    tNumpyMatrixComponents = [None] * numpyImage.ndim
    for i in range(numpyImage.ndim):
        v = [0] * numpyImage.ndim
        v[i] = 1
        tNumpyMatrixComponents[i] = transform(input_image, v) - tNumpyOrigin
    numpyTransform = np.vstack(tNumpyMatrixComponents).dot(np.diag(1 / numpySpacing))

    # define necessary image metadata in header
    header = {
        "origin": numpyOrigin,
        "spacing": numpySpacing,
        "transform": numpyTransform,
    }

    return numpyImage, header

def cal_corr_conv(L):
    L = torch.flatten(L, start_dim=1)
    G = torch.matmul(L, L.T)
    G = torch.triu(G, diagonal=1)
    return torch.sqrt(torch.sum(G ** 2))


def corregularization(model):
    out = 0
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            out += cal_corr_conv(m.weight)

    return out


def rot_flip(x):    # N, C, X, Y, Z
    flipxyz = (torch.where(torch.rand(3) < 0.5)[0] + 2).tolist()    # 2, 3, 4 random extraction
    permxyz = [0, 1] + (torch.randperm(3) + 2).tolist()             # [0, 1, perm(2, 3, 4)]                        # 
    x = x.flip(dims=flipxyz)
    x = x.permute(permxyz)
    
    return x


def rot_flip_yz(x):    # N, C, X, Y, Z
    flipyz = (torch.where(torch.rand(2) < 0.5)[0] + 3).tolist()    # 3, 4 random extraction
    permyz = [0, 1, 2] + (torch.randperm(2) + 3).tolist()             # [0, 1, 2, perm(3, 4)]                        # 
    x = x.flip(dims=flipyz)
    x = x.permute(permyz)
    
    return x

def rot_flip_xy(x):    # N, C, X, Y, Z
    flipyz = (torch.where(torch.rand(3) < 0.5)[0] + 2).tolist()    # 2, 3 random extraction
    permyz = [0, 1] + (torch.randperm(2) + 2).tolist() + [4]             # [0, 1, perm(2, 3), 4]                        # 
    x = x.flip(dims=flipyz)
    x = x.permute(permyz)
    
    return x


def rot_flip_yz_2(x1, x2):    # N, C, X, Y, Z
    
    flipyz = (torch.where(torch.rand(2) < 0.5)[0] + 3).tolist()    # 3, 4 random extraction
    permyz = [0, 1, 2] + (torch.randperm(2) + 3).tolist()             # [0, 1, 2, perm(3, 4)]                        # 
    x1 = x1.flip(dims=flipyz)
    x2 = x2.flip(dims=flipyz)
    x1 = x1.permute(permyz)
    x2 = x2.permute(permyz)
    
    return x1, x2