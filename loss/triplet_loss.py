import torch.nn as nn
import numpy as np


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.margin = margin
        self.mnist_prob_dict = {}
        for x in range(10):
            tmp = np.ones(10)
            tmp[x] = 0
            tmp /= 9
            self.mnist_prob_dict[x] = tmp

    def __call__(self, pred, feats_dict, clss):
        loss = 0.
        bucket = [x for x in feats_dict.keys()]
        for p_feat, c in zip(pred, clss):
            tmp_c = c.cpu().item()
            rand_c = np.random.choice(bucket, 1, p=self.mnist_prob_dict[tmp_c])[0]
            t_feat = feats_dict[tmp_c]
            rand_feat = feats_dict[rand_c]
            tmp_loss = self.loss_fn(p_feat, t_feat) - self.loss_fn(p_feat, rand_feat) + self.margin
            loss += tmp_loss
        loss /= pred.shape[0]
        return loss

if __name__ == "__main__":

    prob_dict = {}
    for x in range(10):
        tmp = np.ones(10)
        tmp[x] = 0

    print(prob_dict)
    a = np.random.choice([x for x in range(10)], 10, p=prob_dict[9])
    print(a)