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

from losses import *

from utils import *
import glob
           
class CrossCSDFirst(nn.Module):
    def __init__(self, args, device, image_channels):
        super(CrossCSDFirst, self).__init__()

        self.device = device
        self.weight_init = args.weight_init
        self.norm = args.norm
        self.opt_clus = -1
        self.layer = args.layer

        self.encoder_source = Encoder(image_channels, norm=args.norm)
        self.decoder_source = Decoder(image_channels, args.layer)

        self.encoder_target = Encoder(image_channels, norm=args.norm)
        self.decoder_target = Decoder(image_channels, args.layer)

    
        self.discriminator_source = DiscriminatorD()
        self.discriminator_target = DiscriminatorD()
        
        initialize_weights(self, self.weight_init)

        self.optimizer_source = optim.Adam(list(self.encoder_source.parameters()) + list(self.decoder_source.parameters()) + list(self.encoder_target.parameters()), lr=args.learning_rate)
        self.scheduler_source = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_source, 'max', patience=args.k2)

        self.optimizer_target = optim.Adam(list(self.encoder_source.parameters()) + list(self.encoder_target.parameters()) + list(self.decoder_target.parameters()), lr=args.learning_rate)
        self.scheduler_target = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_target, 'max', patience=args.k2)

        self.l1_distance = nn.L1Loss().to(device)
        self.cluster_loss = ClusterLoss().to(device)
        self.gen_loss = GeneratorLoss()

        self.optimizer_disc_source = optim.Adam(self.discriminator_source.parameters(), lr=args.learning_rate*0.1)
        self.scheduler_disc_source = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_disc_source, 'max', patience=args.k2)

        self.optimizer_disc_target = optim.Adam(self.discriminator_target.parameters(), lr=args.learning_rate*0.1)
        self.scheduler_disc_target = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_disc_target, 'max', patience=args.k2)

        self.disc_loss = DiscriminatorLoss()

  
    def forward_eval(self, x_s, x_t):
        features_s = self.encoder_source(x_s)
        features_t = self.encoder_target(x_t)
        rec_s = self.decoder_source(features_s[self.layer])
        rec_t = self.decoder_target(features_t[self.layer])

        fake_image_t = self.decoder_target(features_s[self.layer]) # style from tareget domain, content of source domains
        fake_image_s = self.decoder_source(features_t[self.layer])# style from source domain, content of target domains
        
        fake_features_s = self.encoder_source(fake_image_s) 
        fake_features_t = self.encoder_target(fake_image_t)
        cross_rec_s = self.decoder_source(fake_features_t[self.layer]) # should be similar to features_s
        cross_rec_t = self.decoder_target(fake_features_s[self.layer]) # should be similar to features_t

        images_dict = {'Source/image': x_s, 'Target/image': x_t, 'Source/Reconstructed': rec_s, 'Target/Reconstructed': rec_t, 
                       'Source/Cross_reconstructed': cross_rec_s, 'Target/Cross_reconstructed': cross_rec_t, 'Source/fake': fake_image_s, 'Target/fake': fake_image_t}

        return images_dict

    def extract_feats(self, x_s):
        features_s = self.encoder_source(x_s)
        fake_image_t = self.decoder_target(features_s[self.layer])
        fake_features_t = self.encoder_target(fake_image_t)
        return fake_features_t[self.layer]
    
    def extract_feats_true(self, x_t):
        features_t = self.encoder_target(x_t)
        return features_t[self.layer]

    def forward(self, x_s, x_t):
        features_s = self.encoder_source(x_s)
        features_t = self.encoder_target(x_t)
        rec_s = self.decoder_source(features_s[self.layer])
        rec_t = self.decoder_target(features_t[self.layer])

        fake_image_t = self.decoder_target(features_s[self.layer]) # style from tareget domain, content of source domains
        fake_image_s = self.decoder_source(features_t[self.layer]) #style from source domain, content from target domain
        
        fake_features_s = self.encoder_source(fake_image_s)
        fake_features_t = self.encoder_target(fake_image_t)
        cross_rec_s = self.decoder_source(fake_features_t[self.layer])
        cross_rec_t = self.decoder_target(fake_features_s[self.layer])

        results = {"rec_s": rec_s, "cross_img_s": cross_rec_s, "fake_img_s": fake_image_s, "rec_t": rec_t, "cross_img_t": cross_rec_t, "fake_img_t": fake_image_t, 
                    "feats_s": features_s[self.layer], "feats_t": features_t[self.layer], "fake_feats_t": fake_features_t[self.layer]}
        return results


    def update(self, img_s, img_t):
        results = self.forward(img_s, img_t)

        # Discriminators
        prob_true_source_is_true = self.discriminator_source(img_s.detach())
        prob_fake_source_is_true = self.discriminator_source(results["fake_img_s"].detach())
        prob_true_target_is_true = self.discriminator_target(img_t.detach())
        prob_fake_target_is_true = self.discriminator_target(results["fake_img_t"].detach())
        
        # update Discriminators
        disc_loss_s = self.disc_loss(prob_true_source_is_true, prob_fake_source_is_true) 
        disc_loss_t = self.disc_loss(prob_true_target_is_true, prob_fake_target_is_true)

        # print("update discriminator source")
        self.optimizer_disc_source.zero_grad()
        disc_loss_s.backward()
        nn.utils.clip_grad_value_(self.discriminator_source.parameters(), 0.1)
        self.optimizer_disc_source.step()

        # print("update discriminator target")
        self.optimizer_disc_target.zero_grad()
        disc_loss_t.backward()
        nn.utils.clip_grad_value_(self.discriminator_target.parameters(), 0.1)
        self.optimizer_disc_target.step()

        # update source
        prob_fake_source_is_true_gen = self.discriminator_source(results["fake_img_s"])
        gen_loss_s = self.gen_loss(prob_fake_source_is_true_gen)

        reco_loss_s = self.l1_distance(results["rec_s"], img_s)
        cross_reco_loss_s = self.l1_distance(results["cross_img_s"], img_s)

        batch_loss_s = cross_reco_loss_s + gen_loss_s
        # print("update source")
        self.optimizer_source.zero_grad()
        batch_loss_s.backward()
        nn.utils.clip_grad_value_(self.parameters(), 0.1)
        self.optimizer_source.step()
        
        # update target
        results = self.forward(img_s, img_t)

        prob_fake_target_is_true_gen = self.discriminator_target(results["fake_img_t"])
        gen_loss_t = self.gen_loss(prob_fake_target_is_true_gen)

        reco_loss_t = self.l1_distance(results["rec_t"], img_t)
        cross_reco_loss_t = self.l1_distance(results["cross_img_t"], img_t)

        batch_loss_t = cross_reco_loss_t + gen_loss_t
        
        # print("update target")
        self.optimizer_target.zero_grad()
        batch_loss_t.backward()
        nn.utils.clip_grad_value_(self.parameters(), 0.1)
        self.optimizer_target.step()

        batch_loss_it = batch_loss_s.item() + batch_loss_t.item()


        losses = {'loss/source/batch_loss': batch_loss_s.item(), 'loss/source/reco_loss': reco_loss_s.item(), 
                    'loss/source/gen_loss': gen_loss_s.item(), 'loss/source/cross_reco_loss': cross_reco_loss_s.item(),
                    'loss/target/gen_loss': gen_loss_t.item(), 'loss/target/cross_reco_loss': cross_reco_loss_t.item(), 
                    'loss/target/batch_loss': batch_loss_t.item(),'loss/target/reco_loss': reco_loss_t.item(),
                    'loss/source/disc_loss': disc_loss_s.item(),  'loss/target/disc_loss': disc_loss_t.item(), 'loss/total': batch_loss_it}
        return losses
    
    def save(self, save_dir, epoch, score):
        state = {
                'discriminator_S': self.discriminator_source.state_dict(),
                'discriminator_T': self.discriminator_target.state_dict(),
                'encoder_S': self.encoder_source.state_dict(),
                'encoder_T': self.encoder_target.state_dict(),
                'decoder_S': self.decoder_source.state_dict(),
                'decoder_T': self.decoder_target.state_dict(),}
        
        file_name = os.path.join(save_dir, f"ep_{epoch}_loss_{score}.pth")
        torch.save(state, file_name)

    def resume(self, model_file):
        checkpoint = torch.load(model_file)
        self.encoder_source.load_state_dict(checkpoint["encoder_S"]) 
        self.encoder_target.load_state_dict(checkpoint["encoder_T"])
        self.decoder_source.load_state_dict(checkpoint["decoder_S"])
        self.decoder_target.load_state_dict(checkpoint["decoder_T"])
        self.discriminator_source.load_state_dict(checkpoint["discriminator_S"])
        self.discriminator_target.load_state_dict(checkpoint["discriminator_T"])


class CrossCSD(nn.Module):
    def __init__(self, args, device, image_channels, num_classes, vMF_kappa, fold_nr):
        super(CrossCSD, self).__init__()

        self.pretrain = args.pretrain
        self.device = device
        self.num_classes = num_classes
        self.weight_init = args.weight_init
        self.init_method = args.init
        self.opt_clus = -1
        self.layer = args.layer
        self.vc_num  = args.vc_num
        self.true_clu_loss = args.true_clu_loss
        self.fold = fold_nr

        #self.pretrained_model_dir = os.path.join(args.cp, f'pretrain/fold_{fold_nr}/')

        self.activation_layer = ActivationLayer(vMF_kappa)

        self.encoder_source = Encoder(image_channels)
        self.decoder_source = Decoder(image_channels, args.layer)

        self.encoder_target = Encoder(image_channels)
        self.decoder_target = Decoder(image_channels, args.layer)

        self.segmentor = Segmentor(num_classes, args.vc_num, args.layer)
        self.discriminator_source = DiscriminatorD()
        self.discriminator_target = DiscriminatorD()
        
        self.initialize_model()

        self.optimizer_source = optim.Adam(list(self.encoder_source.parameters()) + list(self.decoder_source.parameters()) + list(self.encoder_target.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
        self.scheduler_source = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_source, 'max', patience=args.k2)

        self.optimizer_target = optim.Adam(list(self.encoder_source.parameters()) + list(self.encoder_target.parameters()) + list(self.decoder_target.parameters()) + list(self.conv1o1.parameters())+ list(self.segmentor.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
        self.scheduler_target = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_target, 'max', patience=args.k2)

        self.l1_distance = nn.L1Loss().to(device)
        self.cluster_loss = ClusterLoss().to(device)
    
        self.dice_loss = DiceLoss(include_background=False, softmax=True).to(device)

        self.gen_loss = GeneratorLoss()

        self.optimizer_disc_source = optim.Adam(self.discriminator_source.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        self.optimizer_disc_target = optim.Adam(self.discriminator_target.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        self.disc_loss_s = DiscriminatorLoss()
        self.disc_loss_t = DiscriminatorLoss()

    
    def initialize_model(self):
        initialize_weights(self, self.weight_init)
        
        if self.init_method == "pretrain_true":
            print("WITH PRE INITIALIZED TRUE KERNELS")
            kern_dir = os.path.join(self.pretrained_model_dir, f'kernels_true/dictionary_{self.vc_num}.pickle')
            print(kern_dir)
            weights = getVmfKernels(kern_dir, self.device)
        elif self.init_method == "pretrain_fake":
            print("WITH PRE INITIALIZED FALSE KERNELS")
            kern_dir = os.path.join(self.pretrained_model_dir, f'kernels_false/dictionary_{self.vc_num}.pickle')
            print(kern_dir)
            weights = getVmfKernels(kern_dir, self.device)
        else:
            weights = torch.zeros([self.vc_num, 64, 1, 1]).type(torch.FloatTensor)
            nn.init.xavier_normal_(weights)

        self.conv1o1 = Conv1o1Layer(weights, self.device)
    

    def comp_layer_forward(self, features):
        kernels = self.conv1o1.weight
        vc_activations = self.conv1o1(features[self.layer]) # fp*uk
        vmf_activations = self.activation_layer(vc_activations) # L
        norm_vmf_activations = torch.zeros_like(vmf_activations)
        norm_vmf_activations = norm_vmf_activations.to(self.device)
        for i in range(vmf_activations.size(0)):
            norm_vmf_activations[i, :, :, :] = F.normalize(vmf_activations[i, :, :, :], p=1, dim=0)
        self.vmf_activations = norm_vmf_activations
        self.vc_activations = vc_activations
        return norm_vmf_activations, kernels

    def forward_eval(self, x_s, x_t, x_s_label, x_t_label):
        features_s = self.encoder_source(x_s)
        features_t = self.encoder_target(x_t)
        rec_s = self.decoder_source(features_s[self.layer])
        rec_t = self.decoder_target(features_t[self.layer])

        fake_image_t = self.decoder_target(features_s[self.layer]) # style from tareget domain, content of source domains
        fake_image_s = self.decoder_source(features_t[self.layer])# style from source domain, content of target domains
        
        fake_features_s = self.encoder_source(fake_image_s) 
        fake_features_t = self.encoder_target(fake_image_t)
        cross_rec_s = self.decoder_source(fake_features_t[self.layer]) # should be similar to features_s
        cross_rec_t = self.decoder_target(fake_features_s[self.layer]) # should be similar to features_ts

        cross_rec_loss_s = self.l1_distance(cross_rec_s, x_s)
        cross_rec_loss_t = self.l1_distance(cross_rec_t, x_t)

        com_features_t, _ = self.comp_layer_forward(features_t)
        com_fake_features_t, _ = self.comp_layer_forward(fake_features_t)

        pre_seg_t = self.segmentor(com_features_t)
        pre_fake_seg_t = self.segmentor(com_fake_features_t)

        compact_pred_t = torch.argmax(pre_seg_t, dim=1).unsqueeze(1)
        compact_pred_fake_t = torch.argmax(pre_fake_seg_t, dim=1).unsqueeze(1)

        dsc_target = dice(x_t_label, pre_seg_t, self.num_classes)
        dsc_target_total = torch.mean(dsc_target[1:])
        dsc_target_fake = dice(x_s_label, pre_fake_seg_t, self.num_classes)
        dsc_target_fake_total = torch.mean(dsc_target_fake[1:])

        assert round(x_s.meta["pixdim"][1], 3) == round(x_s.meta["pixdim"][2], 3)
        assert round(x_t.meta["pixdim"][1], 3) == round(x_t.meta["pixdim"][2], 3)

        assd_target = assd(x_t_label, pre_seg_t, self.num_classes, pix_dim=x_t.meta["pixdim"][1])
        assd_target_fake = assd(x_s_label, pre_fake_seg_t, self.num_classes, pix_dim=x_s.meta["pixdim"][1])

        metrics_dict = {'Source/total': cross_rec_loss_s.item(), 'Target/total': cross_rec_loss_t.item(),
                        'Target/DSC': dsc_target_total, 'Target/DSC_fake': dsc_target_fake_total, 'Target/DSC_0': dsc_target[0], 'Target/DSC_1': dsc_target[1],
                        'Target/DSC_fake_0': dsc_target_fake[0], 'Target/DSC_fake_1': dsc_target_fake[1], 'Target/assd': assd_target.item(), 'Target/assd_fake': assd_target_fake.item()}

        images_dict = {'Source/image': x_s, 'Target/image': x_t, 'Target/Cross_reconstructed': cross_rec_t, 
                       'Source/Cross_reconstructed': cross_rec_s, 'Target/fake': fake_image_t, 
                       'Target/Predicted_seg': compact_pred_t, 'Target/Fake_predicted_seg': compact_pred_fake_t, 'Source/label': x_s_label, 'Target/label': x_t_label}



        visuals_dict = {'Target': com_features_t, 'Fake_Target': com_fake_features_t}
        return metrics_dict, images_dict, visuals_dict
    
    def forward_test(self, x_t):
        features_t = self.encoder_target(x_t)
        com_features_t, _ = self.comp_layer_forward(features_t)
        pre_seg_t = self.segmentor(com_features_t)
        compact_pred_t = torch.argmax(pre_seg_t, dim=1).unsqueeze(1)

        return com_features_t, compact_pred_t


    def forward(self, x_s, x_t):
        features_s = self.encoder_source(x_s)
        features_t = self.encoder_target(x_t)
        rec_s = self.decoder_source(features_s[self.layer])
        rec_t = self.decoder_target(features_t[self.layer])
        # print("SHAPES")
        # for item in features_t:
        #     print('features shape: ', item.shape)

        fake_image_t = self.decoder_target(features_s[self.layer]) # style from tareget domain, content of source domains
        fake_image_s = self.decoder_source(features_t[self.layer]) #style from source domain, content from target domain
        
        fake_features_s = self.encoder_source(fake_image_s)
        fake_features_t = self.encoder_target(fake_image_t)
        cross_rec_s = self.decoder_source(fake_features_t[self.layer])
        cross_rec_t = self.decoder_target(fake_features_s[self.layer])

        #com_features_s, kernels = self.comp_layer_forward(features_s)
        com_fake_features_t, kernels = self.comp_layer_forward(fake_features_t)

        #pre_seg_s = self.segmentor(com_features_s)
        pre_fake_seg_t = self.segmentor(com_fake_features_t)

        results = {"rec_s": rec_s, "cross_img_s": cross_rec_s, "fake_img_s": fake_image_s, "rec_t": rec_t, "cross_img_t": cross_rec_t, "fake_img_t": fake_image_t, 
                    "feats_s": features_s[self.layer], "feats_t": features_t[self.layer], "fake_feats_t": fake_features_t[self.layer], "pre_fake_seg_t": pre_fake_seg_t,  "kernels": kernels} # "pre_seg_s": pre_seg_s,
        return results


    def update(self, img_s, label_s, img_t, epoch):
        results = self.forward(img_s, img_t)

        # Discriminators
        prob_true_source_is_true = self.discriminator_source(img_s.detach())
        prob_fake_source_is_true = self.discriminator_source(results["fake_img_s"].detach())
        prob_true_target_is_true = self.discriminator_target(img_t.detach())
        prob_fake_target_is_true = self.discriminator_target(results["fake_img_t"].detach())
        
        # update Discriminators
        disc_loss_s = self.disc_loss_s(prob_true_source_is_true, prob_fake_source_is_true) 
        disc_loss_t = self.disc_loss_t(prob_true_target_is_true, prob_fake_target_is_true)

        # print("update discriminator source")
        self.optimizer_disc_source.zero_grad()
        disc_loss_s.backward()
        nn.utils.clip_grad_value_(self.discriminator_source.parameters(), 0.1)
        self.optimizer_disc_source.step()

        # print("update discriminator target")
        self.optimizer_disc_target.zero_grad()
        disc_loss_t.backward()
        nn.utils.clip_grad_value_(self.discriminator_target.parameters(), 0.1)
        self.optimizer_disc_target.step()

        # update source
        prob_fake_source_is_true_gen = self.discriminator_source(results["fake_img_s"])
        gen_loss_s = self.gen_loss(prob_fake_source_is_true_gen)

        # reco_loss_s = self.l1_distance(results["rec_s"], img_s)
        cross_reco_loss_s = self.l1_distance(results["cross_img_s"], img_s)

        batch_loss_s = cross_reco_loss_s + gen_loss_s #+ reco_loss_s 

        # print("update source")
        self.optimizer_source.zero_grad()
        batch_loss_s.backward()
        nn.utils.clip_grad_value_(self.parameters(), 0.1)
        self.optimizer_source.step()
        
        # update target
        results = self.forward(img_s, img_t)

        prob_fake_target_is_true_gen = self.discriminator_target(results["fake_img_t"])
        gen_loss_t = self.gen_loss(prob_fake_target_is_true_gen)

        # reco_loss_t = self.l1_distance(results["rec_t"], img_t)
        cross_reco_loss_t = self.l1_distance(results["cross_img_t"], img_t.detach())

        if self.true_clu_loss:
            clu_loss_t = self.cluster_loss(results["feats_t"].detach(), results["kernels"])
        else:
            clu_loss_t = self.cluster_loss(results["fake_feats_t"].detach(), results["kernels"])
       
        label_s_oh = F.one_hot(label_s.long().squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2)
        dice_loss_t = self.dice_loss(results["pre_fake_seg_t"], label_s_oh)

        # batch_loss_t = 2 * cross_reco_loss_t + gen_loss_t + dice_loss_t + 0.3 * clu_loss_t # + fake_clu_loss_t 
        batch_loss_t = cross_reco_loss_t + gen_loss_t + dice_loss_t + clu_loss_t # + fake_clu_loss_t 
        

        # print("update target")
        self.optimizer_target.zero_grad()
        batch_loss_t.backward()
        nn.utils.clip_grad_value_(self.parameters(), 0.1)
        self.optimizer_target.step()

        batch_loss_it = batch_loss_s.item() + batch_loss_t.item()


        losses = {'loss/source/batch_loss': batch_loss_s.item(), #'loss/source/reco_loss': reco_loss_s.item(), 
                    #'loss/source/cluster_loss': clu_loss_s.item(), 'loss/source/dice_loss': dice_loss_s.item(),
                    'loss/source/gen_loss': gen_loss_s.item(), 'loss/source/cross_reco_loss': cross_reco_loss_s.item(),
                    #'loss/source/cross_cluster_loss': fake_clu_loss_s.item(), 
                    'loss/target/gen_loss': gen_loss_t.item(), 'loss/target/cross_reco_loss': cross_reco_loss_t.item(), 
                    'loss/target/batch_loss': batch_loss_t.item(),#'loss/target/reco_loss': reco_loss_t.item(),
                    f'loss/target/{self.true_clu_loss}_cluster_loss': clu_loss_t.item(),
                    'loss/target/dice_loss': dice_loss_t.item(),
                    #'loss/target/fake_cluster_loss': fake_clu_loss_t.item(),
                    'loss/source/disc_loss': disc_loss_s.item(),  'loss/target/disc_loss': disc_loss_t.item(), 'loss/total': batch_loss_it,}
                    # 'loss/target/gen_loss_content': gen_loss_co/ntent.item()}
        
        
        return losses
    

    def old_update(self, img_s, label_s, img_t, epoch):
       
        results = self.forward(img_s.detach(), img_t.detach())

        # Discriminators
        prob_true_source_is_true = self.discriminator_source(img_s.detach())
        prob_fake_source_is_true = self.discriminator_source(results["fake_img_s"].detach())
        prob_true_target_is_true = self.discriminator_target(img_t.detach())
        prob_fake_target_is_true = self.discriminator_target(results["fake_img_t"].detach())
        
        # update Discriminators
        disc_loss_s = self.disc_loss_s(prob_true_source_is_true, prob_fake_source_is_true) 
        disc_loss_t = self.disc_loss_t(prob_true_target_is_true, prob_fake_target_is_true)

        # print("update discriminator source")
        self.optimizer_disc_source.zero_grad()
        disc_loss_s.backward()
        nn.utils.clip_grad_value_(self.discriminator_source.parameters(), 0.1)
        self.optimizer_disc_source.step()

        # print("update discriminator target")
        self.optimizer_disc_target.zero_grad()
        disc_loss_t.backward()
        nn.utils.clip_grad_value_(self.discriminator_target.parameters(), 0.1)
        self.optimizer_disc_target.step()


        # reco_loss_s = self.l1_distance(results["rec_s"], img_s)
        cross_reco_loss_s = self.l1_distance(results["cross_img_s"], img_s.detach())
        # reco_loss_t = self.l1_distance(results["rec_t"], img_t)
        cross_reco_loss_t = self.l1_distance(results["cross_img_t"], img_t.detach())

        prob_fake_source_is_true_gen = self.discriminator_source(results["fake_img_s"])
        gen_loss_s = self.gen_loss(prob_fake_source_is_true_gen)

        prob_fake_target_is_true_gen = self.discriminator_target(results["fake_img_t"])
        gen_loss_t = self.gen_loss(prob_fake_target_is_true_gen)


        batch_loss_s = cross_reco_loss_s + gen_loss_t #+ reco_loss_s + reco_loss_t cross_reco_loss_t + gen_loss_s + 

        # print("update source")
        self.optimizer_source.zero_grad()
        batch_loss_s.backward()
        nn.utils.clip_grad_value_(self.parameters(), 0.1)
        self.optimizer_source.step()
        
        
        if epoch > 50:
            results = self.forward(img_s.detach(), img_t.detach())
            
            if self.true_clu_loss:
                clu_loss_t = self.cluster_loss(results["feats_t"].detach(), results["kernels"])
            else:
                clu_loss_t = self.cluster_loss(results["fake_feats_t"].detach(), results["kernels"])
        
            label_s_oh = F.one_hot(label_s.long().squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2)
            dice_loss_t = self.dice_loss(results["pre_fake_seg_t"], label_s_oh)

            # batch_loss_t = 2 * cross_reco_loss_t + gen_loss_t + dice_loss_t + 0.3 * clu_loss_t # + fake_clu_loss_t 
            batch_loss_t = dice_loss_t + 0.5 * clu_loss_t # + fake_clu_loss_t 
            

            # print("update target")
            self.optimizer_target.zero_grad()
            batch_loss_t.backward()
            nn.utils.clip_grad_value_(self.parameters(), 0.1)
            self.optimizer_target.step()

            batch_loss_it = batch_loss_t.item()


            losses = {'loss/source/batch_loss': batch_loss_s.item(),# 'loss/source/reco_loss': reco_loss_s.item(), 
                        #'loss/source/gen_loss': gen_loss_s.item(), 
                        'loss/source/cross_reco_loss': cross_reco_loss_s.item(),
                        'loss/target/gen_loss': gen_loss_t.item(), 
                        #'loss/target/cross_reco_loss': cross_reco_loss_t.item(), 
                        #'loss/target/reco_loss': reco_loss_t.item(),
                         'loss/target/cluster_loss': clu_loss_t.item(),
                        'loss/target/dice_loss': dice_loss_t.item(), 'loss/target/batch_loss': batch_loss_t.item(),
                        'loss/source/disc_loss': disc_loss_s.item(),  'loss/target/disc_loss': disc_loss_t.item(), 'loss/total': batch_loss_it}
            
        
        else:
            losses = {'loss/source/batch_loss': batch_loss_s.item(), #'loss/source/reco_loss': reco_loss_s.item(), 
                        # 'loss/source/gen_loss': gen_loss_s.item(), 
                        'loss/source/cross_reco_loss': cross_reco_loss_s.item(),
                        'loss/target/gen_loss': gen_loss_t.item(),
                        #   'loss/target/cross_reco_loss': cross_reco_loss_t.item(), 
                        #'loss/target/reco_loss': reco_loss_t.item(),
                        'loss/source/disc_loss': disc_loss_s.item(),  'loss/target/disc_loss': disc_loss_t.item(), 'loss/total': batch_loss_s.item()}
        
        return losses
            
    
    def resume(self, model_file, cpu = False):
        if cpu:
            checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_file)
        self.load_state_dict(checkpoint)

