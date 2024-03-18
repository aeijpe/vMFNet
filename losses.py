import torch
import torch.nn as nn
import torch.nn.functional as F


class KlDiv(nn.Module):
    def __init__(self, red='batchmean'):
        super(KlDiv, self).__init__()
        self.lossfunc = nn.KLDivLoss(reduction=red)

    def forward(self, comp_act_t, comp_act_s):
        kl_loss = 0

        comp_act_s = comp_act_s.permute(1, 0, 2, 3)
        comp_act_t = comp_act_t.permute(1, 0, 2, 3)
        it = 0
        for i in range(comp_act_s.shape[0]):
            
            kl_loss_int = self.lossfunc(comp_act_t[i].log(), comp_act_s[i])
            kl_loss += kl_loss_int
        
        return kl_loss / comp_act_s.shape[0]
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