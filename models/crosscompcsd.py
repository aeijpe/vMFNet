import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode
import cv2
import random
from models.encoder import *
from models.decoder import *
from models.segmentor import *
from models.discriminator import *
from models.weight_init import *
from models.compcsd import *
from composition.model import *
from composition.helpers import *

from composition.losses import ClusterLoss
from monai.losses import DiceLoss
from losses import *

from metrics import dice


class CrossCSD(nn.Module):
    def __init__(self, args, device, image_channels, num_classes, vMF_kappa):
        super(CrossCSD, self).__init__()

        self.device = device
        self.num_classes = num_classes
        self.cpt = args.cpt
        self.cps = args.cps
        self.weight_init = args.weight_init
        self.content_disc = args.content_disc
        self.with_kl_loss = args.with_kl_loss
        self.init_method = args.init
        self.vc_num_seg = args.vc_num_seg

        self.rec_module_source = CompCSDRec(device, image_channels, args.layer, args.vc_num, vMF_kappa)
        self.rec_module_target = CompCSDRec(device, image_channels, args.layer, args.vc_num, vMF_kappa)
        self.segmentor = Segmentor(num_classes, args.vc_num_seg, args.layer)

        if self.content_disc:
            self.discriminator = Discriminator(input_channels=args.vc_num_seg).to(device)
            self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=args.learning_rate*0.1)
            self.scheduler_disc = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_disc, 'max', patience=args.k2)
            self.gen_loss = GeneratorLoss()
            self.disc_loss = DiscriminatorLoss()
        
        self.initialize()

        self.optimizer_source = optim.Adam(list(self.rec_module_source.parameters()) + list(self.segmentor.parameters()), lr=args.learning_rate)
        self.scheduler_source = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_source, 'max', patience=args.k2)

        self.optimizer_target = optim.Adam(self.rec_module_target.parameters(), lr=args.learning_rate)
        self.scheduler_target = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_target, 'max', patience=args.k2)

        self.l1_distance_s = nn.L1Loss().to(device)
        self.cluster_loss_s = ClusterLoss()
        self.dice_loss = DiceLoss(softmax=True)

        self.l1_distance_t = nn.L1Loss().to(device)
        self.cluster_loss_t = ClusterLoss()

        if self.with_kl_loss:
            self.kl_diff_loss = KlDiv(red='batchmean').to(device)
            self.kl_dif_lr = 0.005 
            lr_kl_diff = args.learning_rate * self.kl_dif_lr
            self.optimizer_content = optim.Adam(list(self.rec_module_source.parameters()) + list(self.rec_module_target.parameters()), lr=lr_kl_diff)
            self.scheduler_content = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_content, 'max', patience=args.k2)
        




    def initialize(self):
        if self.init_method == "pretrain":
            print("initialize source module with pretrained weights")
            # Initialize the source module
            self.rec_module_source.initialize(self.cps, self.weight_init)
            # initialize the target module
            print("initialize target module with pretrained weights")
            self.rec_module_target.initialize(self.cpt, self.weight_init)
            initialize_weights(self.segmentor, self.weight_init)
        
            # initialize the segmentor
        elif self.init_method == "xavier":
            initialize_weights(self.rec_module_source, self.weight_init)
            initialize_weights(self.rec_module_target, self.weight_init)
            self.rec_module_source.get_xavier_kernels()
            self.rec_module_target.get_xavier_kernels()
            initialize_weights(self.segmentor, self.weight_init)
        
        if self.content_disc:
            initialize_weights(self.discriminator, self.weight_init)


    
    def forward_eval(self, x_s, x_t, x_s_label, x_t_label):
        rec_s, features_s, kernels_s, norm_vmf_activations_s = self.rec_module_source(x_s)
        pre_seg_s = self.segmentor(norm_vmf_activations_s[:, :self.vc_num_seg, :, :])

        gt_oh_s = F.one_hot(x_s_label.long().squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2)
        reco_loss_s = self.l1_distance_s(rec_s, x_s)
        clu_loss_s = self.cluster_loss_s(features_s.detach(), kernels_s) 
        dice_loss_s = self.dice_loss(pre_seg_s, gt_oh_s)

        compact_pred_s = torch.argmax(pre_seg_s, dim=1).unsqueeze(1)
        dsc_source = dice(x_s_label, compact_pred_s, self.num_classes)

        rec_t, features_t, kernels_t, norm_vmf_activations_t = self.rec_module_target(x_t)
        pre_seg_t = self.segmentor(norm_vmf_activations_t[:, :self.vc_num_seg, :, :])

        gt_oh_t = F.one_hot(x_t_label.long().squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2)
        reco_loss_t = self.l1_distance_t(rec_t, x_t)
        clu_loss_t = self.cluster_loss_t(features_t.detach(), kernels_t)
        dice_loss_t = self.dice_loss(pre_seg_t, gt_oh_t)

        compact_pred_t = torch.argmax(pre_seg_t, dim=1).unsqueeze(1)
        dsc_target = dice(x_t_label, compact_pred_t, self.num_classes)


        metrics_dict = {'Source/reco_loss': reco_loss_s.item(), 'Source/cluster_loss': clu_loss_s.item(), 'Source/dice_loss': dice_loss_s.item(),
                        'Target/reco_loss': reco_loss_t.item(), 'Target/cluster_loss': clu_loss_t.item(), 'Target/dice_loss': dice_loss_t.item(),
                         'DSC_Target': dsc_target, 'DSC_Source': dsc_source} #'kl_loss': kl_loss.item(),

        images_dict = {'Source_image': x_s, 'Target_image': x_t, 'Reconstructed_source': rec_s, 'Reconstructed_target': rec_t, 
                       'Segmentation_source': compact_pred_s, 'Segmentation_target': compact_pred_t, 'Source_label': x_s_label, 'Target_label': x_t_label}
        
        visuals_dict = {'Source': norm_vmf_activations_s, 'Target': norm_vmf_activations_t}

        return metrics_dict, images_dict, visuals_dict
    

    def update_normal(self, img_s, label_s, img_t):
        rec_s, features_s, kernels_s, norm_vmf_activations_s = self.rec_module_source(img_s)
        pre_seg_s = self.segmentor(norm_vmf_activations_s[:, :self.vc_num_seg, :, :])

        reco_loss_s = self.l1_distance_s(rec_s, img_s)
        clu_loss_s = self.cluster_loss_s(features_s.detach(), kernels_s) 
        gt_oh_s = F.one_hot(label_s.long().squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2)
        dice_loss_s = self.dice_loss(pre_seg_s, gt_oh_s)

        batch_loss_s = reco_loss_s + clu_loss_s + dice_loss_s

        self.optimizer_source.zero_grad()
        batch_loss_s.backward()
        nn.utils.clip_grad_value_(self.rec_module_source.parameters(), 0.1)
        self.optimizer_source.step()

        rec_t, features_t, kernels_t, norm_vmf_activations_t = self.rec_module_target(img_t)
        reco_loss_t = self.l1_distance_t(rec_t, img_t)
        clu_loss_t = self.cluster_loss_t(features_t.detach(), kernels_t)
     
        batch_loss_t = reco_loss_t + clu_loss_t 

        self.optimizer_target.zero_grad()
        batch_loss_t.backward()
        nn.utils.clip_grad_value_(self.rec_module_target.parameters(), 0.1)
        self.optimizer_target.step()

        batch_loss_it = batch_loss_s.item() + batch_loss_t.item()

        losses = {'loss/source/batch_loss': batch_loss_s.item(), 'loss/source/reco_loss': reco_loss_s.item(), 
                    'loss/source/cluster_loss': clu_loss_s.item(), 'loss/source/dice_loss': dice_loss_s.item(),
                    'loss/target/batch_loss': batch_loss_t.item(),'loss/target/reco_loss': reco_loss_t.item(),
                    'loss/target/cluster_loss': clu_loss_t.item(), 'loss/total': batch_loss_it}
    
        return losses
    
    def update_kl_d(self, img_s, label_s, img_t):
        losses = self.update_normal(img_s, label_s, img_t)

        norm_vmf_activations_s_kl, _, _ = self.rec_module_source.get_activations(img_s)
        norm_vmf_activations_t_kl, _, _ = self.rec_module_target.get_activations(img_t)
    
        kl_loss = self.kl_diff_loss(norm_vmf_activations_t_kl[:, :self.vc_num_seg, :, :], norm_vmf_activations_s_kl[:, :self.vc_num_seg, :, :])

        self.optimizer_content.zero_grad()
        kl_loss.backward()
        nn.utils.clip_grad_value_(self.parameters(), 0.1)
        self.optimizer_content.step()

        true_kl_loss = self.kl_dif_lr  * kl_loss.item()
        batch_loss_it = losses['loss/total'] + true_kl_loss
    
        # Update losses
        losses['loss/kl_loss'] =  true_kl_loss 
        losses['loss/total'] = batch_loss_it

        return losses
    
    def update_disc(self, img_s, label_s, img_t):
        rec_s, features_s, kernels_s, norm_vmf_activations_s = self.rec_module_source(img_s)
    
        prob_source_is_true = self.discriminator(norm_vmf_activations_s[:, :self.vc_num_seg, :, :]) 
        pre_seg_s = self.segmentor(norm_vmf_activations_s[:, :self.vc_num_seg, :, :])


        reco_loss_s = self.l1_distance_s(rec_s, img_s)
        clu_loss_s = self.cluster_loss_s(features_s.detach(), kernels_s) 
        gt_oh_s = F.one_hot(label_s.long().squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2)
        dice_loss_s = self.dice_loss(pre_seg_s, gt_oh_s)
        gen_loss_s = self.gen_loss(prob_source_is_true)

        batch_loss_s = reco_loss_s + clu_loss_s + dice_loss_s + 2 * gen_loss_s

        self.optimizer_source.zero_grad()
        batch_loss_s.backward()
        nn.utils.clip_grad_value_(self.rec_module_source.parameters(), 0.1)
        self.optimizer_source.step()

        rec_t, features_t, kernels_t, norm_vmf_activations_t = self.rec_module_target(img_t)
        reco_loss_t = self.l1_distance_t(rec_t, img_t)
        clu_loss_t = self.cluster_loss_t(features_t.detach(), kernels_t)
        prob_target_is_true = self.discriminator(norm_vmf_activations_t[:, :self.vc_num_seg, :, :])
        gen_loss_t = self.gen_loss(prob_target_is_true)
     
        batch_loss_t = reco_loss_t + clu_loss_t + 2 * gen_loss_t

        self.optimizer_target.zero_grad()
        batch_loss_t.backward()
        nn.utils.clip_grad_value_(self.rec_module_target.parameters(), 0.1)
        self.optimizer_target.step()

        batch_loss_it = batch_loss_s.item() + batch_loss_t.item()

        # update discriminator
        prob_source_is_true_d = self.discriminator(norm_vmf_activations_s[:, :self.vc_num_seg, :, :].detach())
        prob_target_is_true_d = self.discriminator(norm_vmf_activations_t[:, :self.vc_num_seg, :, :].detach())
        disc_loss = self.disc_loss(prob_source_is_true_d, prob_target_is_true_d)
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        nn.utils.clip_grad_value_(self.discriminator.parameters(), 0.1)
        self.optimizer_disc.step()


        losses = {'loss/source/batch_loss': batch_loss_s.item(), 'loss/source/reco_loss': reco_loss_s.item(), 
                    'loss/source/cluster_loss': clu_loss_s.item(), 'loss/source/dice_loss': dice_loss_s.item(),
                    'loss/source/gen_loss': gen_loss_s.item(), 'loss/target/gen_loss': gen_loss_t.item(),
                    'loss/target/batch_loss': batch_loss_t.item(),'loss/target/reco_loss': reco_loss_t.item(),
                    'loss/target/cluster_loss': clu_loss_t.item(), 'loss/disc_loss': disc_loss.item(), 'loss/total': batch_loss_it}
        
        return losses
      


    def update(self, x_s, m_s, x_t):
        if self.with_kl_loss:
            batch_loss_it = self.update_kl_d(x_s, m_s, x_t)
            return batch_loss_it
        elif self.content_disc:
            batch_loss_it = self.update_disc(x_s, m_s, x_t)
            return batch_loss_it
        else:
            batch_loss_it = self.update_normal(x_s, m_s, x_t)
            return batch_loss_it

            
