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

    save_dir = opt.save_dir
    project_name = opt.project_name
    model_name = opt.model_name
    save_interval = opt.save_interval

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
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0003, momentum=0.9)
    lr_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 500, 700], 0.9)
    loss_fn = LossFunction(classes=len(train_loader.dataset.classes), label_smoothing=label_smoothing)

    save_dir = increment_path(Path(save_dir) / project_name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_save_path = os.path.join(save_dir, model_name + ".pt")
    last_save_path = os.path.join(save_dir, model_name + "_last.pt")
    log_save_path = os.path.join(save_dir, model_name + "_log.csv")

    best_loss = 100.
    best_acc = 0.
    best_model_wts = copy.deepcopy(model.state_dict())

    for e in range(start_epoch, end_epoch + 1):
        print(f"\n--- Epoch: {e} / {opt.end_epoch}")
        query_img_dict = train_loader.dataset.query_imgs
        query_feat_dict = get_query_feat(query_img_dict, model, device)

        time.sleep(0.5)
        train_loss = train(model, optimizer, train_loader, loss_fn, device, query_feat_dict, sum_weight)
        time.sleep(0.5)
        print(f"train loss(cls + triplet): {train_loss[0]}")
        print(f"train accuracy: {train_loss[1]: .4f}")

        time.sleep(0.5)
        valid_loss = evaluate(model, valid_loader, loss_fn, device, query_feat_dict, sum_weight)
        time.sleep(0.5)
        print(f"valid loss(cls + triplet): {valid_loss[0]}")
        print(f"valid accuracy: {valid_loss[1]: .4f}")

        if os.path.isfile(log_save_path):
            with open(log_save_path, "r") as f:
                reader = csv.reader(f)
                logs = list(reader)
                logs.append([e] + [x for x in train_loss] + [x for x in valid_loss] + [optimizer.param_groups[0]["lr"]])
            with open(log_save_path, "w") as f:
                writer = csv.writer(f)
                for log in logs:
                    writer.writerow(log)
        else:
            with open(log_save_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow([e] + [x for x in train_loss] + [x for x in valid_loss] + [optimizer.param_groups[0]["lr"]])

        if valid_loss[1] > best_acc:
            best_acc = valid_loss[1]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, best_save_path)

        if e % save_interval == 0 or e == end_epoch:
            torch.save(model.state_dict(), last_save_path)

        lr_sch.step()


def train(model, optimizer, dataloader, loss_fn, device, query_feat_dict, sum_weight):
    model.train()
    train_loss = 0.
    train_acc = 0.

    pbar = tqdm(dataloader)
    for path, img, img0, label in pbar:
        img = img.to(device).float() / 255.
        label = label.to(device)

        optimizer.zero_grad()
        score, feat = model(img)
        cls_loss, triplet_loss = loss_fn(score, label, feat, query_feat_dict)
        loss = cls_loss * (1 - sum_weight) + triplet_loss * sum_weight
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_cls = torch.argmax(score, dim=-1)
            correct = label == pred_cls
            acc = torch.sum(torch.ones_like(label)[correct]) / label.shape[0]

        train_loss += loss.item()
        train_acc += acc.item()
    train_loss /= len(pbar)
    train_acc /= len(pbar)
    return train_loss, train_acc


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, query_feat_dict, sum_weight):
    model.train()
    train_loss = 0.
    train_acc = 0.

    pbar = tqdm(dataloader)
    for path, img, img0, label in pbar:
        img = img.to(device).float() / 255.
        label = label.to(device)

        score, feat = model(img)
        cls_loss, triplet_loss = loss_fn(score, label, feat, query_feat_dict)
        loss = cls_loss * (1 - sum_weight) + triplet_loss * sum_weight

        with torch.no_grad():
            pred_cls = torch.argmax(score, dim=-1)
            correct = label == pred_cls
            acc = torch.sum(torch.ones_like(label)[correct]) / label.shape[0]

        train_loss += loss.item()
        train_acc += acc.item()
    train_loss /= len(pbar)
    train_acc /= len(pbar)
    return train_loss, train_acc


def parse_opt():
    parser = argparse.ArgumentParser()

    pre_weights = None
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

    parser.add_argument("--save-dir", type=str, default="weights")
    parser.add_argument("--project-name", type=str, default="exp")
    parser.add_argument("--model-name", type=str, default="effnet2")
    parser.add_argument("--save-interval", type=int, default=25)

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
