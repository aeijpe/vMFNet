import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from eval import eval_vmfnet_uns
from mmwhs_dataloader import MMWHS
from models.compcsd import CompCSDMM
from models.crosscompcsd import CrossCSD
from composition.losses import ClusterLoss
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import pytorch_lightning as pl

import torch.nn.functional as F
import glob

from eval import eval_vmfnet_mm
from losses import KlDiv



def get_args():
    usage_text = (
        "vMFNet Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-e','--epochs', type= int, default=100, help='Number of epochs')
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
    parser.add_argument('--data_dir_t',  type=str, default='../data/other/CT_withGT_proc/annotated/', help='The name of the checkpoints.')
    parser.add_argument('--data_dir_s',  type=str, default='../data/other/MR_withGT_proc/annotated/', help='The name of the checkpoints.')
    parser.add_argument('--k_folds', type= int, default=5, help='Cross validation')
    parser.add_argument('--pred', type=str, default='MYO', help='Segmentation task')
    parser.add_argument('--with_kl_loss', action='store_true') 
    parser.add_argument('--content_disc', action='store_true') 
    parser.add_argument('--vc_num_seg', type=int,  default=2, help='Kernel/distributions amount as input for the segmentation model')
    parser.add_argument('--init', type=str, default='pretrain', help='Initialization method') # pretrain (original), xavier, cross.
    

    return parser.parse_args()


def train_net(train_loader, val_loader, fold, device, args, len_train_data, num_classes):
    dir_checkpoint = os.path.join(args.cp, args.name)
    save_dir = os.path.join(dir_checkpoint, f'fold_{fold}')
    os.makedirs(save_dir, exist_ok=True)

    #Model selection and initialization
    model = CrossCSD(args, device, 1, num_classes, vMF_kappa=30)
    model.to(device)

    log_dir = os.path.join('logs', os.path.join(args.name, 'fold_{}'.format(fold)))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_score = 0

    global_step = 0
    print("Training fold: ", fold)

    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len_train_data, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            model.train()
            for img_s, label_s, img_t, label_t in train_loader:
                img_s = img_s.to(device)
                img_t = img_t.to(device)
                label_s = label_s.to(device)

                losses = model.update(img_s, label_s, img_t)
                pbar.set_postfix(**{'loss (batch)': losses["loss/source/batch_loss"]})
                pbar.update(img_t.shape[0])

                for key, value in losses.items():
                    writer.add_scalar(f'{key}', value, global_step)

                global_step += 1
    
            
            if epoch % 5 == 0:
                metrics_dict, images_dict, visuals_dict = eval_vmfnet_mm(model, val_loader, device)

                for key, value in metrics_dict.items():
                    writer.add_scalar(f'Val_metrics/{key}', value, epoch)
                
                for key, value in images_dict.items():
                    writer.add_images(f'Val_images/{key}', value, epoch, dataformats='NCHW')
                
                for key, value in visuals_dict.items():
                    for i in range(args.vc_num):
                        writer.add_images(f'Val_visuals/{key}_{i+1}', value[:,i,:,:].unsqueeze(1), epoch, dataformats='NCHW')

                #on which parameter are we going to choose the best model??  --> For now cheating
                if metrics_dict["DSC_Target"] > best_score:
                    best_score = metrics_dict["DSC_Target"]
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
                    
                    torch.save(model.state_dict(), os.path.join(save_dir, f'CP_epoch_{epoch}_DSC_T_{metrics_dict["DSC_Target"]}.pth'))
                    logging.info('Checkpoint saved !')


            
    writer.close()


def main(args):
    pl.seed_everything(args.seed)

    if args.pred == "MYO":
        print("MYO prediction")
        labels = [1, 0, 0, 0, 0, 0, 0]
        num_classes = 2
    elif args.pred =="three":
        print("three")
        labels = [1, 0, 2, 0, 3, 0, 0] # 2 is LVC and 3 is RVC
        num_classes = 4
    else:
        print("ALL")
        labels = [1, 2, 3, 4, 5, 6, 7]
        num_classes = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = range(0,18)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0
    
 
    for fold_train, fold_val in kf.split(cases):
        print('Train fold:', fold_train)
        print('Val fold:', fold_val)
        print("loading train data")
        dataset_train = MMWHS(args, labels, fold_train)
        train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
        print("loading val data")
        dataset_val = MMWHS(args, labels, fold_val) 
        val_loader = DataLoader(dataset_val, batch_size=args.bs, drop_last=True, num_workers=4)
        len_train_data = dataset_train.__len__()

        train_net(train_loader, val_loader, fold, device, args, len_train_data, num_classes) 
        fold += 1
        print("For now one fold")
        break




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)
