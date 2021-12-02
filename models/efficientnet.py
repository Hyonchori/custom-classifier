import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=False, dropout=0.2):
        super().__init__()
        effnet = efficientnet_b0(pretrained)
        self.feature_extractor = effnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes)
        )
        self.out_act = nn.Softmax(dim=-1)

    def __call__(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        score = self.out_act(self.classifier(feat))
        if self.training:
            return score, feat
        else:
            return score


if __name__ == "__main__":
    import torch
    model = EfficientClassifier(num_classes=10)
    sample = torch.randn((32, 3, 128, 128))
    score, feat = model(sample)
    print(score.shape, feat.shape)
    print(score[0])
