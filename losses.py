import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


class KlDiv(nn.Module):
    def __init__(self, red='batchmean'):
        super(KlDiv, self).__init__()
        self.lossfunc = nn.KLDivLoss(reduction=red)

    def forward(self, comp_act_t, comp_act_s):
        comp_act_t_perm = F.log_softmax(comp_act_t, dim=1)
        comp_act_s_perm = F.softmax(comp_act_s, dim=1)

        kl_loss_int = self.lossfunc(comp_act_t_perm, comp_act_s_perm)
        if torch.isnan(kl_loss_int).any():
            print("kl loss is nan at ")
           
        
        return kl_loss_int
        #     if not torch.isnan(kl_loss_int).any():
        #         kl_loss += kl_loss_int
        #         it += 1
                
        # if it == 0:
        #     return 0
        # else:
        #     return kl_loss / it
    

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, prob_real_is_real, prob_fake_is_real):
        loss = (torch.mean(torch.pow(prob_real_is_real - 1, 2)) + torch.mean(torch.pow(prob_fake_is_real, 2))) * 0.5
        return loss
    
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
    
    def forward(self, prob_fake_repr_is_real):
        lsgan_loss = torch.mean(torch.pow(prob_fake_repr_is_real - 1, 2)) * 0.5
        return lsgan_loss
    


class DiscriminatorLoss2(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss2, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, prob_real_is_real, prob_fake_is_real):
        real_labels = torch.ones_like(prob_real_is_real)
        fake_labels = torch.zeros_like(prob_fake_is_real)

        real_loss = self.criterion(prob_real_is_real, real_labels)
        fake_loss = self.criterion(prob_fake_is_real, fake_labels)

        loss = real_loss + fake_loss
        return loss
    
class GeneratorLoss2(nn.Module):
    def __init__(self):
        super(GeneratorLoss2, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, prob_fake_repr_is_real):
        real_labels = torch.ones_like(prob_fake_repr_is_real)

        gen_loss = self.criterion(prob_fake_repr_is_real, real_labels)
        return gen_loss

class DiceLossMC(nn.Module):
    def __init__(self, n_classes):
        super(DiceLossMC, self).__init__()
        self.n_classes = n_classes 
        self.loss = DiceLoss()
    
    def forward(self, pred, gt):
        loss = 0
        pred_sm = F.softmax(pred, dim=1)
        for i in range(self.n_classes):
            # skip the background
            if i == 0 :
                continue
            loss_i = self.loss(pred_sm[:, i, :, :].unsqueeze(1), gt[:, i, :, :].unsqueeze(1))
            loss += loss_i
        
        return loss/3
