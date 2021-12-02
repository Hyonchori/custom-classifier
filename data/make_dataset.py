import os

import cv2
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split


def make_clf_dirs(data_set, train_root, out_root, out_dir="train"):
    for img_name, img_label in zip(data_set["filen_name"], data_set["label"]):
        img_path = os.path.join(train_root, img_name)
        img = cv2.imread(img_path)

        label_dir = os.path.join(out_root, out_dir, str(img_label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        out_path = os.path.join(label_dir, img_name)
        cv2.imwrite(out_path, img)


def make_query_gallary(img_root, query_prob=0.2):
    labels = [x for x in sorted(os.listdir(img_root)) if os.path.isdir(os.path.join(img_root, x))]
    print(labels)
    for l in labels:
        l_path = os.path.join(img_root, l)
        file_names = [x for x in sorted(os.listdir(l_path)) if x.endswith(".png")]
        gallary_set, query_set = train_test_split(file_names, test_size=query_prob, random_state=42)
        print(len(gallary_set), len(query_set))
        for img_name in gallary_set:
            img_path = os.path.join(l_path, img_name)
            img = cv2.imread(img_path)
            label_dir = os.path.join(l_path, "gallery")
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            out_path = os.path.join(label_dir, img_name)
            cv2.imwrite(out_path, img)

        for img_name in query_set:
            img_path = os.path.join(l_path, img_name)
            img = cv2.imread(img_path)
            label_dir = os.path.join(l_path, "query")
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            out_path = os.path.join(label_dir, img_name)
            cv2.imwrite(out_path, img)


def make_train_valid(train_root, train_csv, out_root, save=False):
    data = pd.read_csv(train_csv)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_idx, valid_idx in split.split(data, data["label"]):
        train_set = data.loc[train_idx]
        valid_set = data.loc[valid_idx]

    if save:
        make_clf_dirs(train_set, train_root, out_root, "train")
        make_clf_dirs(valid_set, train_root, out_root, "valid")


if __name__ == "__main__":
    root = "/media/jhc/4AD250EDD250DEAF/dataset/dacon/mnist"
    train_root = os.path.join(root, "train")
    test_root = os.path.join(root, "test")
    csv_path = os.path.join(root, "train_data.csv")

    # make dataset for classification from original dataset
    '''out_root = "/media/jhc/4AD250EDD250DEAF/dataset/dacon/mnist/basic"
    make_train_valid(train_root, csv_path, out_root, save=True)'''

    # make dataset for triplet training from classfication dataset
    clf_root = "/media/daton/D6A88B27A88B0569/dataset/mnist/query_gallary/train"
    make_query_gallary(clf_root)