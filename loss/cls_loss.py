import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
           https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == "__main__":
    import torch
    p = torch.tensor([[0.1, 0.8, 0.1]], requires_grad=True)
    targets = torch.tensor([1])

    loss_fn = LabelSmoothingLoss(3, smoothing=0.1)
    loss = loss_fn(p, targets)
    print(loss)
