# Copyright (c) 2022 vios-s

import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from eval import eval_vmfnet
from dataloader import CHAOS_single, MMWHS_single
from models.compcsd import CompCSD
from composition.losses import ClusterLoss
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import torch.nn.functional as F
import glob
import numpy as np
from utils import *

from monai.losses import DiceLoss



def get_args():
    usage_text = (
        "vMFNet Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('--epochs', type= int, default=200, help='Number of epochs')
    parser.add_argument('--seed', default=42, type=int,help='Seed to use for reproducing results') # --> their default was 14
    parser.add_argument('--bs', type= int, default=4, help='Number of inputs per batch')
    parser.add_argument('--cp', type=str, default='checkpoints/', help='The name of the checkpoints.')
    parser.add_argument('--cps', type=str, default='checkpoints/single_MR', help='The name of the checkpoints for source.')
    parser.add_argument('--cpt', type=str, default='checkpoints/single_CT', help='The name of the checkpoints for target.')

    parser.add_argument('--name', type=str, default='test_MM_KLD', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--k1', type=int,  default=40, help='When the learning rate starts decaying')
    parser.add_argument('--k2', type=int,  default=4, help='Check decay learning')
    parser.add_argument('--layer', type=int,  default=8, help='layer from which the deepf eatures are obtained')
    parser.add_argument('--vc_num', type=int,  default=12, help='Kernel/distributions amount')
    parser.add_argument('--data_dir',  type=str, default='../data/other/CT_withGT_proc/annotated/', help='The name of the checkpoints.')
    parser.add_argument('--k_folds', type= int, default=5, help='Cross validation')
    parser.add_argument('--pred', type=str, default='MYO', help='Segmentation task')
    parser.add_argument('--init', type=str, default='xavier', help='Initialization method') # pretrain (original), xavier, cross.
    parser.add_argument('--norm', type=str, default="Batch")
    parser.add_argument('--data_type', default='MMWHS', type=str, help='Baseline used') 

    return parser.parse_args()


# Train function per fold
def train_net(train_loader, val_loader, fold, device, args, num_classes, len_train_data):
    dir_checkpoint = os.path.join(args.cp, args.name)

    #Model selection and initialization
    model = CompCSD(device, 1, args.layer, args.vc_num, num_classes=num_classes, z_length=8, vMF_kappa=30, init=args.init)
    model.initialize(dir_checkpoint, args.weight_init)
    model.to(device)

    #metrics initialization
    l1_distance = nn.L1Loss().to(device)
    cluster_loss = ClusterLoss()
    dice_loss = DiceLoss(softmax=True)

    #optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.k2)

    log_dir = os.path.join('logs', os.path.join(args.name, 'fold_{}'.format(fold)))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    best_val_score = 0
    print("Training fold: ", fold)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len_train_data, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            model.train()
            for img_s, label_s in train_loader:
                img_s = img_s.to(device)
                label_s = label_s.to(device)

                # Forward pass
                rec_s, pre_seg_s, features_s, kernels_s, L_visuals_s = model(img_s)

                # Calculate losses
                labels_oh = F.one_hot(label_s.long().squeeze(1), num_classes).permute(0, 3, 1, 2)
                loss_dice_s = dice_loss(pre_seg_s, labels_oh) 
                reco_loss_s = l1_distance(rec_s, img_s)
                clu_loss_s = cluster_loss(features_s.detach(), kernels_s)
                batch_loss_s = reco_loss_s + clu_loss_s  + loss_dice_s
    
                # Optimize
                optimizer.zero_grad()
                batch_loss_s.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                # TB logging
                writer.add_scalar('loss/batch_loss_s', batch_loss_s.item(), global_step)
                writer.add_scalar('loss/reco_loss_s', reco_loss_s.item(), global_step)
                writer.add_scalar('loss/loss_dice_s', loss_dice_s.item(), global_step)
                writer.add_scalar('loss/cluster_loss_s', clu_loss_s.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': batch_loss_s.item()})

                pbar.update(img_s.shape[0])

                global_step += 1

            if optimizer.param_groups[0]['lr'] <= 2e-8:
                print('Converge')

            # Validation step
            if epoch % 5 == 0:
                dsc_classes, show_imgs, show_rec, show_true, show_pred, show_vis = eval_vmfnet(model, val_loader, device)

                # TB logging
                for i, item in enumerate(dsc_classes):
                    writer.add_scalar(f'Val_metrics/dice_class_{i}', item, epoch)
                
                dsc_classes = dsc_classes.cpu()
                total_dsc = np.mean(np.array(dsc_classes)[1:])
                
                if epoch % 50 == 0:
                    writer.add_images(f'Val_images/img', show_imgs, epoch, dataformats='NCHW')
                    writer.add_images(f'Val_images/rec', show_rec, epoch, dataformats='NCHW')
                    writer.add_images(f'Val_images/label', show_true, epoch, dataformats='NCHW')
                    writer.add_images(f'Val_images/pred', show_pred, epoch, dataformats='NCHW')
                
                scheduler.step(total_dsc)

                # Save best model during training
                if total_dsc > best_val_score:
                    best_val_score = total_dsc
                    print("Epoch checkpoint")
                    save_dir = os.path.join(dir_checkpoint, f'fold_{fold}')
                    os.makedirs(save_dir, exist_ok=True)

                    print(f"--- Remove old model before saving new one ---")
                    # Define the pattern to search for existing model files in the directory
                    existing_model_files = glob.glob(f"{save_dir}/*.pth")
                    # Delete each found model file
                    for file_path in existing_model_files:
                        try:
                            os.remove(file_path)
                            print(f"Deleted old model file: {file_path}")
                        except OSError as e:
                            print(f"Error deleting file {file_path}: {e}")
                    
                    
                    torch.save(model.state_dict(), os.path.join(save_dir, f'CP_epoch_{epoch}_dice_{best_val_score}.pth'))
                    logging.info('Checkpoint saved !')

            
    writer.close()


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0

    labels, num_classes = get_labels(args.pred)

    # Get Dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS_single
    elif args.data_type == "CHAOS":
        dataset_type = CHAOS_single
    else:
        raise ValueError(f"Data type {args.data_type} not supported")
    
    # Cross-validation
    for fold_train, fold_val in kf.split(cases):
        print("loading train data")
        dataset_train = dataset_type(args.data_dir, fold_train, labels)
        train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
        
        print("loading val data")
        dataset_val = dataset_type(args.data_dir, fold_val, labels) 
        val_loader = DataLoader(dataset_val, batch_size=args.bs, drop_last=True, num_workers=4)
        len_train_data = dataset_train.__len__()

        # Train for this fold
        train_net(train_loader, val_loader, fold, device, args, num_classes, len_train_data) 
        fold += 1
    


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)
