
import argparse

import torch
from torch import nn
from torch.autograd import grad
import math


def wgan_QCloss(pred_d_real, pred_d_fake,HStar_real,HStar_fake):
    mean_HStar_real = HStar_real.mean()

    diff = pred_d_fake- HStar_fake
    diffSqu = diff * diff

    loss = 0.5*pow(pred_d_real.mean() - mean_HStar_real,2)
    loss += 0.5*diffSqu.mean()
    return loss


def gradient_penalty_WQC(preds, realImg, fakeImg, KCoef):
    differences = realImg - fakeImg
    gradients = grad(preds, fakeImg,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    differences_norm = differences.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - KCoef*differences_norm)**2).mean()
    gradient_penalty = gradient_penalty/math.sqrt(KCoef)
    return gradient_penalty