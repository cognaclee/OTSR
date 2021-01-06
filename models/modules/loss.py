import torch
import torch.nn as nn
import math


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss 
        elif self.gan_type == 'wgan-qc':
            def wgan_QC(pred_d, target, HStar):
                # target is boolean
                if target:
                    mean_HStar_real = HStar.mean()
                    loss = pow(pred_d.mean() - mean_HStar_real,2)          
                else:
                    diff = pred_d - HStar
                    diffSqu = diff * diff
                    loss = diffSqu.mean()    
                return loss

            self.loss = wgan_QC
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        
        if self.gan_type == 'wgan-qc':
            loss = self.loss(input[0], target_is_real ,input[1])          
        else:
            target_label = self.get_target_label(input, target_is_real)
            loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

class QC_GradientPenaltyLoss(nn.Module):
    def __init__(self):
        super(QC_GradientPenaltyLoss, self).__init__()


    def forward(self, preds, realImg, fakeImg, KCoef):
        differences = realImg - fakeImg
        gradients = torch.autograd.grad(preds, fakeImg,
                         grad_outputs=torch.ones_like(preds),
                         retain_graph=True, create_graph=True)[0]

        gradient_norm = gradients.norm(2, dim=1)
        differences_norm = differences.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - KCoef*differences_norm)**2).mean()/math.sqrt(KCoef)
        return gradient_penalty

