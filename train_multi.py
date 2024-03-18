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
    parser.add_argument('--cps', type=str, default='checkpoints/test', help='The name of the checkpoints for source.')
    parser.add_argument('--cpt', type=str, default='checkpoints/reconstruct_CT', help='The name of the checkpoints for target.')

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

    return parser.parse_args()


def train_net(train_loader, val_loader, fold, device, args, len_train_data, num_classes):
    best_score = 1000000
    dir_checkpoint = os.path.join(args.cp, args.name)
    save_dir = os.path.join(dir_checkpoint, f'fold_{fold}')
    os.makedirs(save_dir, exist_ok=True)
    source_model_name = glob.glob(os.path.join(args.cps, f'fold_{fold}/*.pth'))[0]

    #Model selection and initialization
    model = CompCSDMM(args, device, 1, num_classes, vMF_kappa=30, type_model = 'kldiv')
    model.initialize(source_model_name)
    model.to(device)

    log_dir = os.path.join('logs', os.path.join(args.name, 'fold_{}'.format(fold)))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    print("Training fold: ", fold)


    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len_train_data, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            model.train()
            for img_s, label_s, img_t, label_t in train_loader:
                img_s = img_s.to(device)
                img_t = img_t.to(device)

                batch_loss_it = model.update(img_s, img_t, writer, global_step)
                pbar.set_postfix(**{'loss (batch)': batch_loss_it})
                pbar.update(img_t.shape[0])

                global_step += 1

            #if (epoch + 1) > args.k1 and (epoch + 1) % args.k2 == 0:
            # on which parameter are we going to choose the best model?? 
            if epoch % 5 == 0:
                l1_val, clu_val, dice_score, imgs, rec, true_label, pred_label, L_visual_source, L_visual_target = eval_vmfnet_mm(model, val_loader, device)
                val_score = l1_val + clu_val 
             
                model.scheduler.step(val_score)
                writer.add_scalar('learning_rate', model.optimizer.param_groups[0]['lr'], epoch)

                writer.add_scalar('Val_metrics/l1_dist', l1_val, epoch)
                writer.add_scalar('Val_metrics/cluster_loss', clu_val, epoch)
                writer.add_scalar('Val_metrics/DSC', dice_score, epoch)
                writer.add_scalar('Val_metrics/score', val_score, epoch)

                writer.add_images('Val_images/image_true', imgs, epoch, dataformats='NCHW')
                writer.add_images('Val_images/image_rec', rec, epoch, dataformats='NCHW')
                writer.add_images('Val_images/label_true', true_label, epoch, dataformats='NCHW')
                writer.add_images('Val_images/label_pred', pred_label, epoch, dataformats='NCHW')


                writer.add_images('L_visuals_val/L_1_s', L_visual_source[:,0,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_2_s', L_visual_source[:,1,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_3_s', L_visual_source[:,2,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_4_s', L_visual_source[:,3,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_5_s', L_visual_source[:,4,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_6_s', L_visual_source[:,5,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_7_s', L_visual_source[:,6,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_8_s', L_visual_source[:,7,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_9_s', L_visual_source[:,8,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_10_s', L_visual_source[:,9,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_11_s', L_visual_source[:,10,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_12_s', L_visual_source[:,11,:,:].unsqueeze(1), epoch, dataformats='NCHW')

                writer.add_images('L_visuals_val/L_1_t', L_visual_target[:,0,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_2_t', L_visual_target[:,1,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_3_t', L_visual_target[:,2,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_4_t', L_visual_target[:,3,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_5_t', L_visual_target[:,4,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_6_t', L_visual_target[:,5,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_7_t', L_visual_target[:,6,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_8_t', L_visual_target[:,7,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_9_t', L_visual_target[:,8,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_10_t', L_visual_target[:,9,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_11_t', L_visual_target[:,10,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals_val/L_12_t', L_visual_target[:,11,:,:].unsqueeze(1), epoch, dataformats='NCHW')


                if val_score < best_score:
                    best_score = val_score
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
                    
                    torch.save(model.state_dict(), os.path.join(save_dir, f'CP_epoch_{epoch}_recloss_{val_score}.pth'))
                    logging.info('Checkpoint saved !')


            
    writer.close()


def main(args):
    pl.seed_everything(args.seed)

    if args.pred == "MYO":
        labels = [1, 0, 0, 0, 0, 0, 0]
        num_classes = 2
    else:
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




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)
