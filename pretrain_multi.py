import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from mmwhs_dataloader import MMWHS
from models.crosscompcsd import CrossCSD, CrossCSDFirst
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import pytorch_lightning as pl
import glob
import numpy as np

from eval import eval_vmfnet_mm
from utils import *



def get_args():
    usage_text = (
        "vMFNet Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('--epochs', type= int, default=100, help='Number of epochs')
    parser.add_argument('--seed', default=42, type=int,help='Seed to use for reproducing results') # --> their default was 14
    parser.add_argument('--bs', type= int, default=4, help='Number of inputs per batch')
    parser.add_argument('--cp', type=str, default='checkpoints/', help='The name of the checkpoints.')

    parser.add_argument('--name', type=str, default='test_MM_KLD', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    # parser.add_argument('--k1', type=int,  default=40, help='When the learning rate starts decaying')
    parser.add_argument('--k2', type=int,  default=10, help='Check decay learning')
    parser.add_argument('--layer', type=int,  default=8, help='layer from which the deepf eatures are obtained')
    parser.add_argument('--vc_num', type=int,  default=12, help='Kernel/distributions amount')
    parser.add_argument('--data_dir_t',  type=str, default='../data/other/CT_withGT_proc/annotated/', help='The name of the data dir.')
    parser.add_argument('--data_dir_s',  type=str, default='../data/other/MR_withGT_proc/annotated/', help='The name of the data dir.')
    parser.add_argument('--k_folds', type= int, default=5, help='Cross validation')
    parser.add_argument('--pred', type=str, default='MYO', help='Segmentation task')
   
    parser.add_argument('--vc_num_seg', type=int,  default=12, help='Kernel/distributions amount as input for the segmentation model')
    parser.add_argument('--init', type=str, default='pretrain', help='Initialization method') # pretrain (original), xavier, cross.
    parser.add_argument('--norm', type=str, default="Batch")

    parser.add_argument('--true_clu_loss', action='store_true')  # default is optimization with fake loss
    parser.add_argument('--content_disc', action='store_true')  # with or without content discriminator
    parser.add_argument('--data_type', type=str, default="MMWHS") #MMWHS, RetinalVessel
    

    return parser.parse_args()


def pretrain(args, device, train_loader, save_dir, len_data, fold):

    #Model selection and initialization
    model = CrossCSDFirst(args, device, 1)
    model.to(device)

    log_dir = os.path.join('logs', os.path.join(args.name, f'pretrain/fold_{fold}'))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_score = np.inf

    global_step = 0 

    for epoch in range(args.epochs):
        model.train()
        total_epoch_loss = 0
        with tqdm(total=len_data, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            model.train()
            for img_s, label_s, img_t, label_t in train_loader:
                img_s = img_s.to(device)
                img_t = img_t.to(device)

                losses = model.update(img_s, img_t)
                pbar.set_postfix(**{'loss (batch)': losses["loss/total"]})
                pbar.update(img_t.shape[0])

                for key, value in losses.items():
                    writer.add_scalar(f'{key}', value, global_step)

                global_step += 1
                total_epoch_loss += losses["loss/total"]

            if total_epoch_loss < best_score:
                best_score = total_epoch_loss
                print("Epoch checkpoint")

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
                
                model.save(save_dir, epoch, best_score)
                logging.info('Checkpoint saved !')
            
    writer.close()


def pretrain_k_folds(args, labels, device, dataset_type):
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0
    
    # for fold_train, fold_test_val in kf.split(cases):
    for fold_train_val, fold_test in kf.split(cases):
        dir_checkpoint = os.path.join(args.cp, args.name)
        save_dir = os.path.join(dir_checkpoint, f'pretrain/fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)
        
        dataset_train = dataset_type(args, labels, fold_train_val)
        train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
        len_data = len(dataset_train)
        pretrain(args, device, train_loader, save_dir, len_data, fold)
        fold += 1
      


def main(args):
    pl.seed_everything(args.seed)

    labels, num_classes = get_labels(args.pred)
    # MMWHS Dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS
        in_channels = 1
    elif args.data_type == "RetinalVessel":
        NotImplementedError
        # dataset_type = Retinal_Vessel_single
        # in_channels = 3
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    pretrain_k_folds(args, labels, device, dataset_type)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)
