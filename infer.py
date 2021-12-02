import argparse
import copy
import os
import csv
import time
from pathlib import Path

import cv2
import torch
import torch.optim as optim
from tqdm import tqdm

from data.load_dataset import get_mnist_dataloader
from models.efficientnet import EfficientClassifier
from loss import LossFunction, get_query_feat
from utils.general import increment_path


def main(opt):
    pre_weights = opt.pre_weights
    num_classes = opt.num_classes

    train_root = opt.train_root
    valid_root = opt.valid_root
    label_smoothing = opt.label_smoothing
    sum_weight = opt.sum_weight
    start_epoch = opt.start_epoch
    end_epoch = opt.end_epoch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EfficientClassifier(num_classes=num_classes, pretrained=False)
    if pre_weights is not None:
        if os.path.isfile(pre_weights):
            wts = torch.load(pre_weights)
            model.load_state_dict(wts)
            print("Model is initialized with existing weights!")
        else:
            print("Weight's path is wrong!. Model is just initialized!")
    else:
        print("Model is initialized!")
    model = model.to(device)

    train_loader, valid_loader = get_mnist_dataloader(train_root=train_root, valid_root=valid_root)
    loss_fn = LossFunction(classes=len(train_loader.dataset.classes), label_smoothing=label_smoothing)

    valid_loss = evaluate(model, valid_loader, loss_fn, device, sum_weight)
    print("")
    print(valid_loss)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, sum_weight):
    model.train()
    train_loss = 0.
    train_acc = 0.

    for path, img, img0, label in dataloader:
        img = img.to(device).float() / 255.
        label = label.to(device)

        score, feat = model(img)

        for im0, s, l in zip(img0, score, label):
            print(f"\n--- {l.cpu().numpy()}")
            print(s)
            cv2.imshow("im0", im0.numpy())
            cv2.waitKey(0)

        with torch.no_grad():
            pred_cls = torch.argmax(score, dim=-1)
            correct = label == pred_cls
            acc = torch.sum(torch.ones_like(label)[correct]) / label.shape[0]

        #train_loss += loss.item()
        train_acc += acc.item()
    print(score[0])
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def parse_opt():
    parser = argparse.ArgumentParser()

    pre_weights = "/home/daton/PycharmProjects/pythonProject/custom-classifier/weights/exp2/effnet2_last.pt"
    parser.add_argument("--pre-weights", type=str, default=pre_weights)
    parser.add_argument("--num-classes", type=int, default=10)

    train_root = "/media/daton/D6A88B27A88B0569/dataset/mnist/query_gallery/train"
    valid_root = "/media/daton/D6A88B27A88B0569/dataset/mnist/query_gallery/valid"
    parser.add_argument("--train-root", type=str, default=train_root)
    parser.add_argument("--valid-root", type=str, default=valid_root)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--sum-weight", type=float, default=0.2)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=1000)

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
