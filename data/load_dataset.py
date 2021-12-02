import os

import numpy as np
import cv2
import torch
import albumentations as AT


class DatasetForClassify(torch.utils.data.Dataset):
    def __init__(self,
                 root: str,
                 train: bool=True,
                 subset_num: int=None,
                 transform: AT.core.composition.Compose=None):
        self.root = root
        self.train = train
        self.subset_num = subset_num if subset_num is not None else ""
        self.transform = transform

        self.classes = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
        self.imgs_and_labels, self.query_imgs = self.get_imgs()

    def get_imgs(self):
        imgs_and_labels = []
        query_imgs_dict = {}
        for cls_idx, cls in enumerate(self.classes):
            if self.train:
                cls_dir = os.path.join(self.root, cls, f"gallery{self.subset_num}")
                query_dir = os.path.join(self.root, cls, f"query{self.subset_num}")
                query_imgs = sorted(os.listdir(query_dir))
                query_imgs_dict[cls_idx] = [os.path.join(self.root, cls, x) for x in query_imgs]
            else:
                cls_dir = os.path.join(self.root, cls)
            cls_imgs = sorted(os.listdir(cls_dir))
            cls_paths_and_labels = [(os.path.join(self.root, cls, x), cls_idx) for x in cls_imgs]
            imgs_and_labels += cls_paths_and_labels

        return imgs_and_labels, query_imgs_dict

    def __len__(self):
        return len(self.imgs_and_labels)

    def __getitem__(self, idx):
        target_img = self.imgs_and_labels[idx]
        img_path = target_img[0]
        img_label = target_img[1]

        img0 = cv2.imread(img_path)

        if self.transform is not None:
            transformed = self.transform(image=img0)
            img = transformed["image"]
        else:
            img = img0

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, img_label


train_transform = AT.Compose([
    AT.Rotate(limit=20),
    AT.Affine(translate_percent=(0.1, 0.1)),
    AT.Perspective(),
])


def get_mnist_dataloader(train_root: str = "/media/daton/D6A88B27A88B0569/dataset/mnist/query_gallery/train",
                         valid_root: str = "/media/daton/D6A88B27A88B0569/dataset/mnist/query_gallery/valid",
                         train_transform: AT.core.composition.Compose = train_transform,
                         subset_num: int = None,
                         train_batch: int = 64,
                         valid_batch: int = 64):
    train_dataset = DatasetForClassify(train_root, True, subset_num, train_transform)
    valid_dataset = DatasetForClassify(valid_root, False, subset_num, None)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch,
        shuffle=False
    )
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    train_root = "/media/jhc/4AD250EDD250DEAF/dataset/dacon/mnist/triplet/train"
    valid_root = "/media/jhc/4AD250EDD250DEAF/dataset/dacon/mnist/triplet/valid"

    # Dataset test
    '''train_dataset = DatasetForClassify(train_root, transform=train_transform)
    vaild_dataset = DatasetForClassify(valid_root, train=False)

    print(len(train_dataset), len(vaild_dataset))

    for path, img, img0, img_label in train_dataset:
        print("\n---")
        print(path)
        print(img_label)
        cv2.imshow("img", img.transpose(1, 2, 0))
        cv2.imshow("img0", img0)
        cv2.waitKey(0)'''

    # Dataloader test
    train_dataloader, valid_dataloader = get_mnist_dataloader()
    for path, img, img0, img_label in train_dataloader:
        print(img.shape)
        print(img0.shape)
        for imm, im, label in zip(img, img0, img_label):
            print(label)
            cv2.imshow("imt", imm.numpy().transpose(1, 2, 0))
            cv2.imshow("img", im.numpy())
            cv2.waitKey(0)

    # Query test
    '''train_dataloader, valid_dataloader = get_mnist_dataloader()
    query_imgs_dict = train_dataloader.dataset.query_imgs
    for cls, img_path in query_imgs_dict.items():
        print(cls)
        img = cv2.imread(img_path[0])
        cv2.imshow("img", img)
        cv2.waitKey(0)'''
