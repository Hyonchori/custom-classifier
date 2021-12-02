import sys
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).absolute()
if FILE.parents[0].as_posix() not in sys.path:
    sys.path.append(FILE.parents[0].as_posix())

import cv2
import torch.nn as nn

from cls_loss import LabelSmoothingLoss
from triplet_loss import TripletLoss


class LossFunction():
    def __init__(self, classes=10, label_smoothing=0.15):
        #self.class_loss_fn = LabelSmoothingLoss(classes, label_smoothing)
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.triplet_loss_fn = TripletLoss()

    def __call__(self, p_clss, t_clss, p_feats, feats_dict):
        cls_loss = self.class_loss_fn(p_clss, t_clss)
        triplet_loss = self.triplet_loss_fn(p_feats, feats_dict, t_clss)
        return cls_loss, triplet_loss


@torch.no_grad()
def get_query_feat(query_imgs_dict, model, device):
    model.train()
    query_feats = {}
    for img_cls, img_paths in query_imgs_dict.items():
        img_batch = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img[None]).to(device).float()
            img_batch.append(img)
        img_batch = torch.cat(img_batch)
        _, img_feat = model(img_batch)
        img_feat = torch.mean(img_feat, dim=0)
        query_feats[img_cls] = img_feat
    return query_feats
