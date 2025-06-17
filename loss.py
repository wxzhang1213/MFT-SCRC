import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate


def Con_loss(x, y, temperature=0.5):
    batch = x.size()[0]
    logits = (x @ y.T) / temperature
    targets = torch.arange(len(logits)).to(logits.device)
    loss = (nn.functional.cross_entropy(logits, targets) +
        nn.functional.cross_entropy(logits.T, targets)) / (2 * batch)
    return loss


def Con_Label_Loss(x, y, labels):
    batch = x.size()[0]
    pos_x = x[labels.squeeze() > 0]
    neg_x = x[labels.squeeze() < 0]
    pos_y = y[labels.squeeze() > 0]
    neg_y = y[labels.squeeze() < 0]
    if len(pos_x) > 0 and len(neg_x) > 0:
        positive = torch.exp(Sim(pos_x, pos_y)) + torch.exp(Sim(neg_x, neg_y))
        negative = torch.exp(Sim(pos_x, neg_y)) + torch.exp(Sim(neg_x, pos_y))
        loss = -torch.log((positive + 1e-6) / (positive + negative + 1e-6)) / batch
    else:
        loss = 0.0
    return loss
