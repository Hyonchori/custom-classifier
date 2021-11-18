
import argparse
import os
from pathlib import Path
FILE = Path(__file__).absolute()

from utils.general import Config
from utils.augmentation import build_transforms
from data.dataset import get_data_for_train


def main(opt):
    effnet_config = Config(opt.train_config_path)
    print(effnet_config)

    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)


    train_loader, val_loader, num_query, num_classes = get_data_for_train(effnet_config)


def parse_opt():
    parser = argparse.ArgumentParser()

    train_config_path = f"{FILE.parents[0]}/configs/train/efficientnetv2_train.yaml"
    parser.add_argument("--train-config-path", type=str, default=train_config_path)

    output_dir = f"{FILE.parents[0]}/train_log"
    parser.add_argument("--output-dir", type=str, default=output_dir)



    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
