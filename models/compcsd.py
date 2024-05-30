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
    def __init__(self, device, image_channels, layer, vc_numbers, num_classes, z_length, vMF_kappa, init):
        super(CompCSD, self).__init__()

        self.image_channels = image_channels
        self.layer = layer
        self.z_length = z_length
        self.num_classes = num_classes
        self.vc_num = vc_numbers

        self.activation_layer = ActivationLayer(vMF_kappa)

        self.encoder = Encoder(self.image_channels)
        self.segmentor = Segmentor(self.num_classes, self.vc_num)
        self.decoder = Decoder(self.image_channels)
        self.device = device
        self.init_method = init

    def initialize(self, cp_dir, init="xavier"):
        initialize_weights(self, init)
        if self.init_method == "pretrain":
            enc_dir = os.path.join(cp_dir, 'encoder/UNet.pth')
            kern_dir = os.path.join(cp_dir, 'kernels/dictionary_12.pickle')
            self.load_encoder_weights(enc_dir, self.device)
            self.load_vmf_kernels(kern_dir)
        else:
            self.get_xavier_kernels()

    def get_xavier_kernels(self):
        weights = torch.zeros([self.vc_num, 64, 1, 1]).type(torch.FloatTensor)
        weights = weights.to(self.device)
        nn.init.xavier_normal_(weights)
        self.conv1o1 = Conv1o1Layer(weights, self.device)

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
        pre_seg = self.segmentor(norm_vmf_activations)

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
    def __init__(self, device, image_channels, layer, vc_numbers, vMF_kappa, norm="Batch"):
        super(CompCSDRec, self).__init__()

        self.image_channels = image_channels
        self.layer = layer
        self.vc_num = vc_numbers

        self.activation_layer = ActivationLayer(vMF_kappa)

        self.encoder = Encoder(self.image_channels, norm=norm)
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

    def get_image(self, norm_vmf_act):
        decoding_features = self.compose(norm_vmf_act)
        rec = self.decoder(decoding_features)
        return rec

    
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