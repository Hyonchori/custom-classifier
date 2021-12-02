import argparse
import copy
import os
import csv
import time
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from data.load_dataset import get_mnist_dataloader
from models.efficientnet2 import EfficientNetv2
from loss import LossFunction


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EfficientNetv2(opt.num_classes)
    if opt.weights is not None:
        wts = torch.load(opt.weights)
        model.load_state_dict(wts)
        print("Model is initialized with existing weights!")
    else:
        print("Model is initialized!")
    model = model.to(device)

    train_dataloader, valid_dataloader = get_mnist_dataloader(train_root="/media/daton/D6A88B27A88B0569/dataset/mnist/query_gallery/train",
                                                              valid_root="/media/daton/D6A88B27A88B0569/dataset/mnist/query_gallery/valid")
    loss_fn = LossFunction(classes=len(train_dataloader.dataset.classes), label_smoothing=opt.label_smoothing)
    valid_loss = evaluate(model, valid_dataloader, loss_fn, device, opt.sum_weight)
    time.sleep(0.5)
    print("")
    print(valid_loss)


#@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, sum_weight):
    model.train()
    valid_loss = 0.
    valid_acc = 0.

    pbar = tqdm(dataloader)
    for path, img, img0, label in pbar:
        img = img.to(device).float() / 255.
        label = label.to(device)

        score = model(img)
        #score = model(img)

        with torch.no_grad():
            pred_cls = torch.argmax(score, dim=-1)
            correct = label == pred_cls
            accuracy = torch.sum(torch.ones_like(label)[correct]) / label.shape[0]

        #valid_loss += loss.item()
        valid_acc += accuracy.item()
    print(score)
    valid_loss /= len(pbar)
    valid_acc /= len(pbar)
    return valid_loss, valid_acc


def parse_opt():
    parser = argparse.ArgumentParser()

    pre_weights = "weights/exp10/effnet2_last.pt"
    parser.add_argument("--weights", type=str, default=pre_weights)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.15)
    parser.add_argument("--sum-weight", type=float, default=0.2)

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
