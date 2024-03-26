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
from composition.model import *
from composition.helpers import *

from composition.losses import ClusterLoss
from losses import *

class CompCSD(nn.Module):
    def __init__(self, device, image_channels, layer, vc_numbers, num_classes, z_length, vMF_kappa):
        super(CompCSD, self).__init__()

        self.image_channels = image_channels
        self.layer = layer
        self.z_length = z_length
        self.num_classes = num_classes
        self.vc_num = vc_numbers

        self.activation_layer = ActivationLayer(vMF_kappa)

        self.encoder = Encoder(self.image_channels)
        self.segmentor = Segmentor(self.num_classes, self.vc_num, self.layer)
        self.decoder = Decoder(self.image_channels, self.layer)
        self.device = device

    def initialize(self, cp_dir, init="xavier"):
        initialize_weights(self, init)
        enc_dir = os.path.join(cp_dir, 'encoder/UNet.pth')
        kern_dir = os.path.join(cp_dir, 'kernels/init/dictionary/dictionary_12.pickle')
        print(enc_dir)
        print(kern_dir)
        self.load_encoder_weights(enc_dir, self.device)
        self.load_vmf_kernels(kern_dir)

    def forward(self, x):
        kernels = self.conv1o1.weight
        features = self.encoder(x)
        vc_activations = self.conv1o1(features[self.layer]) # fp*uk
        vmf_activations = self.activation_layer(vc_activations) # L
        norm_vmf_activations = torch.zeros_like(vmf_activations)
        norm_vmf_activations = norm_vmf_activations.to(self.device)
        for i in range(vmf_activations.size(0)):
            norm_vmf_activations[i, :, :, :] = F.normalize(vmf_activations[i, :, :, :], p=1, dim=0)
        self.vmf_activations = norm_vmf_activations
        self.vc_activations = vc_activations
        decoding_features = self.compose(norm_vmf_activations)
        rec = self.decoder(decoding_features)
        pre_seg = self.segmentor(norm_vmf_activations, features)

        return rec, pre_seg, features[self.layer], kernels, norm_vmf_activations
    
    def get_activations(self, x):
        features = self.encoder(x)
        vc_activations = self.conv1o1(features[self.layer]) # fp*uk
        vmf_activations = self.activation_layer(vc_activations) # L
        norm_vmf_activations = torch.zeros_like(vmf_activations)
        norm_vmf_activations = norm_vmf_activations.to(self.device)
        for i in range(vmf_activations.size(0)):
            norm_vmf_activations[i, :, :, :] = F.normalize(vmf_activations[i, :, :, :], p=1, dim=0)
        return norm_vmf_activations, vc_activations, features[self.layer]

    def load_encoder_weights(self, dir_checkpoint, device):
        self.device = device
        pre_trained = torch.load(dir_checkpoint, map_location=self.device) # without unet.
        new = list(pre_trained.items())

        my_model_kvpair = self.encoder.state_dict() # with unet.
        count = 0
        for key in my_model_kvpair:
            layer_name, weights = new[count]
            my_model_kvpair[key] = weights
            count += 1
        self.encoder.load_state_dict(my_model_kvpair)

    def load_vmf_kernels(self, dict_dir):
        weights = getVmfKernels(dict_dir, self.device)
        self.conv1o1 = Conv1o1Layer(weights, self.device)

    def compose(self, vmf_activations):
        kernels = self.conv1o1.weight #[512, 128, 1, 1]
        kernels = kernels.squeeze(2).squeeze(2) # [512, 128]

        features = torch.zeros([vmf_activations.size(0), kernels.size(1), vmf_activations.size(2), vmf_activations.size(3)])
        features = features.to(self.device)

        for k in range(vmf_activations.size(0)):
            single_vmf_activations = vmf_activations[k]
            single_vmf_activations = torch.permute(single_vmf_activations, (1, 2, 0))  # [512, 72, 72]
            feature = torch.matmul(single_vmf_activations, kernels)
            feature = torch.permute(feature, (2, 0, 1))
            features[k, :, :, :] = feature
        return features
    
    def resume(self, model_dir):
        checkpoint = torch.load(model_dir)
        self.load_state_dict(checkpoint)


class CompCSDRec(nn.Module):
    def __init__(self, device, image_channels, layer, vc_numbers, vMF_kappa):
        super(CompCSDRec, self).__init__()

        self.image_channels = image_channels
        self.layer = layer
        self.vc_num = vc_numbers

        self.activation_layer = ActivationLayer(vMF_kappa)

        self.encoder = Encoder(self.image_channels)
        self.decoder = Decoder(self.image_channels, self.layer)
        self.device = device

    def initialize(self, cp_dir, init="xavier"):
        initialize_weights(self, init)
        enc_dir = os.path.join(cp_dir, 'encoder/UNet.pth')
        kern_dir = os.path.join(cp_dir, 'kernels/init/dictionary/dictionary_12.pickle')
        self.load_encoder_weights(enc_dir, self.device)
        self.load_vmf_kernels(kern_dir)

    def forward(self, x):
        kernels = self.conv1o1.weight
        norm_vmf_activations, vc_activations, feats = self.get_activations(x)
        self.vmf_activations = norm_vmf_activations
        self.vc_activations = vc_activations
        decoding_features = self.compose(norm_vmf_activations)
        rec = self.decoder(decoding_features)
        return rec, feats, kernels, norm_vmf_activations
    
    def get_activations(self, x):
        features = self.encoder(x)
        vc_activations = self.conv1o1(features[self.layer]) # fp*uk
        vmf_activations = self.activation_layer(vc_activations) # L
        norm_vmf_activations = torch.zeros_like(vmf_activations)
        norm_vmf_activations = norm_vmf_activations.to(self.device)
        for i in range(vmf_activations.size(0)):
            norm_vmf_activations[i, :, :, :] = F.normalize(vmf_activations[i, :, :, :], p=1, dim=0)
        return norm_vmf_activations, vc_activations, features[self.layer]


    def load_encoder_weights(self, dir_checkpoint, device):
        self.device = device
        pre_trained = torch.load(dir_checkpoint, map_location=self.device) # without unet.
        new = list(pre_trained.items())

        my_model_kvpair = self.encoder.state_dict() # with unet.
        count = 0
        for key in my_model_kvpair:
            layer_name, weights = new[count]
            my_model_kvpair[key] = weights
            count += 1
        self.encoder.load_state_dict(my_model_kvpair)

    def load_vmf_kernels(self, dict_dir):
        weights = getVmfKernels(dict_dir, self.device)
        self.conv1o1 = Conv1o1Layer(weights, self.device)

    
    def get_xavier_kernels(self):
        weights = torch.zeros([self.vc_num, 64, 1, 1]).type(torch.FloatTensor)
        weights = weights.to(self.device)
        nn.init.xavier_normal_(weights)
        self.conv1o1 = Conv1o1Layer(weights, self.device)


    def compose(self, vmf_activations):
        kernels = self.conv1o1.weight #[512, 128, 1, 1]
        kernels = kernels.squeeze(2).squeeze(2) # [512, 128]

        features = torch.zeros([vmf_activations.size(0), kernels.size(1), vmf_activations.size(2), vmf_activations.size(3)])
        features = features.to(self.device)

        for k in range(vmf_activations.size(0)):
            single_vmf_activations = vmf_activations[k]
            single_vmf_activations = torch.permute(single_vmf_activations, (1, 2, 0))  # [512, 72, 72]
            feature = torch.matmul(single_vmf_activations, kernels)
            feature = torch.permute(feature, (2, 0, 1))
            features[k, :, :, :] = feature
        return features
    
    def resume(self, model_dir):
        checkpoint = torch.load(model_dir)
        self.load_state_dict(checkpoint)



class CompCSDMM(nn.Module):
    def __init__(self, args, device, image_channels, num_classes, vMF_kappa, type_model ='kldiv'):
        super(CompCSDMM, self).__init__()

        self.rec_module_source = CompCSD(device, image_channels, args.layer, args.vc_num, num_classes, vMF_kappa)
        self.rec_module_target = CompCSDRec(device, image_channels, args.layer, args.vc_num, vMF_kappa)
        self.optimizer = optim.Adam(list(self.rec_module_source.parameters()) + list(self.rec_module_target.parameters()), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=args.k2)
        self.l1_distance = nn.L1Loss().to(device)
        self.cluster_loss = ClusterLoss()
        self.device = device
        self.num_classes = num_classes
        self.cpt = args.cpt
        self.cps = args.cps
        self.weight_init = args.weight_init
        self.type_mod = type_model

        if type_model == 'kldiv':
            self.kl_diff_loss = KlDiv(red='batchmean').to(device)
        elif type_model == 'disc': 
            self.discriminator = Discriminator()
            self.discriminator.to(device)
            self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.1*(args.learning_rate))
            self.discriminator_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.discriminator_optimizer, 'max', patience=args.k2)
            self.gen_loss = GeneratorLoss()
            self.disc_loss = DiscriminatorLoss()

    def initialize(self, model_file_src):
        # Get pretrained rec module and freeze model
        self.rec_module_source.initialize(self.cps, self.weight_init)
        self.rec_module_source.resume(model_file_src)

        # Freeze this part
        for param in self.rec_module_source.parameters():
            param.requires_grad = False

        # initialize the target module
        self.rec_module_target.initialize(self.cpt, self.weight_init)


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

        batch_loss = reco_loss + 1 * clu_loss + 0.000001 * kl_loss
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
    