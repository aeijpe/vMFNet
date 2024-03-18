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


class CrossCSD(nn.Module):
    def __init__(self, args, device, image_channels, num_classes, vMF_kappa):
        super(CrossCSD, self).__init__()

        self.rec_module_source = CompCSDRec(device, image_channels, args.layer, args.vc_num, vMF_kappa)
        self.rec_module_target = CompCSDRec(device, image_channels, args.layer, args.vc_num, vMF_kappa)
        self.segmentor = Segmentor(num_classes, args.layer)

        self.optimizer_source = optim.Adam(list(self.rec_module_source.parameters()) + list(self.segmentor), lr=args.learning_rate)
        self.scheduler_source = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_source, 'max', patience=args.k2)

        self.optimizer_target = optim.Adam(list(self.rec_module_target.parameters()) + list(self.segmentor), lr=args.learning_rate)
        self.scheduler_target = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_target, 'max', patience=args.k2)

        self.l1_distance = nn.L1Loss().to(device)
        self.cluster_loss = ClusterLoss()
        self.dice_loss = DiceLoss()

        self.device = device
        self.num_classes = num_classes
        self.cpt = args.cpt
        self.cps = args.cps
        self.weight_init = args.weight_init
        # self.type_mod = type_model


    def initialize(self, model_file_src):
        # Initialize the source module
        self.rec_module_source.initialize(self.cps, self.weight_init)
        # initialize the target module
        self.rec_module_target.initialize(self.cpt, self.weight_init)
        initialize_weights(self.segmentor, self.weight_init)


    def forward(self, x_s, x_t):
        rec, features_t, kernels, norm_vmf_activations_t = self.rec_module_target(x_t)
        norm_vmf_activations_s, _, features_s = self.rec_module_source.get_activations(x_s)
        return rec, features_t, kernels, norm_vmf_activations_t, norm_vmf_activations_s, features_s
    
    def forward_eval(self, x_s, x_t):
        rec, features_t, kernels, norm_vmf_activations_t, norm_vmf_activations_s, features_s = self.forward(x_s, x_t)
        pre_seg = self.rec_module_source.segmentor(norm_vmf_activations_t)
        return pre_seg, rec, features_t, kernels, norm_vmf_activations_t, norm_vmf_activations_s, features_s 

    def update_kl_d(self, img_s, img_t, writer, global_step):
        rec, features_t, kernels, norm_vmf_activations_t, norm_vmf_activations_s, features_s = self.forward(img_s, img_t)


        reco_loss = self.l1_distance(rec, img_t)
        clu_loss = self.cluster_loss(features_t.detach(), kernels) 

        kl_loss = self.kl_diff_loss(norm_vmf_activations_t, norm_vmf_activations_s)

        batch_loss = reco_loss + 0.7 * clu_loss + 0.0001 * kl_loss
        batch_loss_it = batch_loss.item()

        self.optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_value_(self.parameters(), 0.1)
        self.optimizer.step()

        writer.add_scalar('loss/batch_loss', batch_loss.item(), global_step)
        writer.add_scalar('loss/reco_loss', reco_loss.item(), global_step)
        writer.add_scalar('loss/cluster_loss', clu_loss.item(), global_step)
        writer.add_scalar('loss/kl_div_loss', kl_loss.item(), global_step)

        if self.optimizer.param_groups[0]['lr'] <= 2e-8:
            print('Converge')

        return batch_loss_it

    def update_disc(self, img_s, img_t, writer, global_step):
        rec, features_t, kernels, norm_vmf_activations_t, norm_vmf_activations_s, features_s = self.forward(img_s, img_t)
        prob_target_is_true = self.discriminator(norm_vmf_activations_t.detach()) # this is 'fake'
        if global_step % 10 == 0:
            prob_target_is_true2 = self.discriminator(norm_vmf_activations_t.detach()) # this is 'fake'
            prob_source_is_true2 = self.discriminator(norm_vmf_activations_s.detach()) # this is 'real'
        reco_loss = self.l1_distance(rec, img_t)
        clu_loss = self.cluster_loss(features_t.detach(), kernels) 

        gen_loss = self.gen_loss(prob_target_is_true)

        batch_loss = reco_loss + clu_loss + 0.1 * gen_loss # hyperparameter tuning???
        batch_loss_it = batch_loss.item()
        self.optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_value_(self.parameters(), 0.1)
        self.optimizer.step()

        if global_step % 10 == 0:
            disc_loss = self.disc_loss(prob_source_is_true2, prob_target_is_true2)
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            nn.utils.clip_grad_value_(self.discriminator.parameters(), 0.1)
            self.discriminator_optimizer.step()
            writer.add_scalar('loss/disc_loss', disc_loss.item(), global_step)

        writer.add_scalar('loss/batch_loss', batch_loss.item(), global_step)
        writer.add_scalar('loss/reco_loss', reco_loss.item(), global_step)
        writer.add_scalar('loss/cluster_loss', clu_loss.item(), global_step)
        writer.add_scalar('loss/gen_loss', gen_loss.item(), global_step)
        


        if self.optimizer.param_groups[0]['lr'] <= 2e-8:
            print('Converge')

        return batch_loss_it

    def update(self, x_s, x_t, writer, global_step):
        if self.type_mod == 'kldiv':
            batch_loss_it = self.update_kl_d(x_s, x_t, writer, global_step)
            return batch_loss_it
        elif self.type_mod == 'disc':
            batch_loss_it = self.update_disc(x_s, x_t, writer, global_step)
            return batch_loss_it
        else:
            print('type nog available')

    def forward_inference(self, x):
        norm_vmf_activations, _, _ = self.rec_module_target.get_activations(x)
        pre_seg = self.rec_module_source.segmentor(norm_vmf_activations)
        return pre_seg, norm_vmf_activations
    