# EfficientNet v2 for classification and feature extraction

import torch
import torch.nn as nn

from building_blocks import Conv, Residual, initialize_weights


class EfficientNet2Base(nn.Module):
    def __init__(self, args):
        super().__init__()
        gate_fn = [True, False]
        filters = [24, 48, 64, 128, 160, 256, 1280]

        base = [Conv(args, 3, filters[0], torch.nn.SiLU(), 3, 2)]
        for i in range(2):
            if i == 0:
                base.append(
                    Residual(args, filters[0], filters[0], 1, 1, gate_fn[0]))
            else:
                base.append(
                    Residual(args, filters[0], filters[0], 1, 1, gate_fn[0]))

        for i in range(4):
            if i == 0:
                base.append(
                    Residual(args, filters[0], filters[1], 2, 4, gate_fn[0]))
            else:
                base.append(
                    Residual(args, filters[1], filters[1], 1, 4, gate_fn[0]))

        for i in range(4):
            if i == 0:
                base.append(
                    Residual(args, filters[1], filters[2], 2, 4, gate_fn[0]))
            else:
                base.append(
                    Residual(args, filters[2], filters[2], 1, 4, gate_fn[0]))

        for i in range(6):
            if i == 0:
                base.append(
                    Residual(args, filters[2], filters[3], 2, 4, gate_fn[1]))
            else:
                base.append(
                    Residual(args, filters[3], filters[3], 1, 4, gate_fn[1]))

        for i in range(9):
            if i == 0:
                base.append(
                    Residual(args, filters[3], filters[4], 1, 6, gate_fn[1]))
            else:
                base.append(
                    Residual(args, filters[4], filters[4], 1, 6, gate_fn[1]))

        for i in range(15):
            if i == 0:
                base.append(
                    Residual(args, filters[4], filters[5], 2, 6, gate_fn[1]))
            else:
                base.append(
                    Residual(args, filters[5], filters[5], 1, 6, gate_fn[1]))
        base.append(Conv(args, filters[5], filters[6], torch.nn.SiLU()))

        self.base = torch.nn.Sequential(*base)
        initialize_weights(self)

    def forward(self, x):
        x = self.base(x)
        return x

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model'].state_dict()

        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class EfficientNetv2(nn.Module):
    def __init__(self, num_classes, neck="bnneck", training=True, args=True, verbose=False):
        super().__init__()

        self.in_planes = 1280
        self.base = EfficientNet2Base(args)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_clssses = num_classes
        self.neck = neck
        self.trainig = training

        if self.neck == "no":
            self.classifier = nn.Linear(self.in_planes, self.num_clssses)
        elif self.neck == "bnneck":
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)
            self.classifier = nn.Linear(self.in_planes, self.num_clssses, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, in_planes, 1, 1)
        global_feat = global_feat.view(
            global_feat.shape[0], -1)  # flatten to (bs, in_planes)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            # normalize for angular softmax
            feat = self.bottleneck(global_feat)

        cls_score = self.classifier(feat).softmax(dim=1)
        if self.training:
            return cls_score, global_feat  # global feature for triplet loss
        else:
            return cls_score

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


if __name__ == "__main__":
    model = EfficientNetv2(255)
    sample = torch.rand((32, 3, 256, 128))
    score, feat = model(sample)
    print(score.shape, feat.shape)

    model.eval()
    score = model(sample)
    print(score.shape)
    pass