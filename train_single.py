import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from eval import eval_vmfnet, eval_vmfnet_mm_sm
from mmwhs_dataloader import MMWHS_single, MMWHS
from models.compcsd import CompCSD
from composition.losses import ClusterLoss
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import pytorch_lightning as pl

from monai.losses import DiceLoss
import torch.nn.functional as F
import glob



def get_args():
    usage_text = (
        "vMFNet Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('--epochs', type= int, default=300, help='Number of epochs')
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
    parser.add_argument('--norm', type=str, default="Batch")

    return parser.parse_args()


def train_net(train_loader, val_loader, fold, device, args, num_classes, len_train_data):
    best_dice = 0
    dir_checkpoint = os.path.join(args.cp, args.name)

    #Model selection and initialization
    model = CompCSD(device, 1, args.layer, args.vc_num, num_classes=num_classes, z_length=8, vMF_kappa=30, init=args.init)
    # num_params = utils.count_parameters(model)
    # print('Model Parameters: ', num_params)
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
    print("Training fold: ", fold)

    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len_train_data, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            model.train()
            for img_s, label_s, img_t, label_t in train_loader:
                img_s = img_s.to(device)
                img_t = img_t.to(device)
                label_s = label_s.to(device)

                rec_s, pre_seg_s, features_s, kernels_s, L_visuals_s = model(img_s)

                gt_oh_s = F.one_hot(label_s.long().squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2)
                loss_dice_s = dice_loss(pre_seg_s, gt_oh_s) # CHECK DIMENSIONS


                reco_loss_s = l1_distance(rec_s, img_s)
                clu_loss_s = cluster_loss(features_s.detach(), kernels_s)

                batch_loss_s = reco_loss_s + clu_loss_s  + loss_dice_s

               

                optimizer.zero_grad()
                batch_loss_s.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                writer.add_scalar('loss/batch_loss_s', batch_loss_s.item(), global_step)
                writer.add_scalar('loss/reco_loss_s', reco_loss_s.item(), global_step)
                writer.add_scalar('loss/loss_dice_s', loss_dice_s.item(), global_step)
                writer.add_scalar('loss/cluster_loss_s', clu_loss_s.item(), global_step)

                # update with target img
                rec_t, pre_seg_t, features_t, kernels_t, L_visuals_t = model(img_t)

                reco_loss_t = l1_distance(rec_t, img_t)
                clu_loss_t = cluster_loss(features_t.detach(), kernels_t)

                batch_loss_t = reco_loss_t + clu_loss_t

                optimizer.zero_grad()
                batch_loss_t.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                writer.add_scalar('loss/batch_loss_t', batch_loss_t.item(), global_step)
                writer.add_scalar('loss/reco_loss_t', reco_loss_t.item(), global_step)
                writer.add_scalar('loss/cluster_loss_t', clu_loss_t.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': batch_loss_s.item() + batch_loss_t.item()})

                pbar.update(img_s.shape[0])

                global_step += 1

            if optimizer.param_groups[0]['lr'] <= 2e-8:
                print('Converge')

            #if (epoch + 1) > args.k1 and (epoch + 1) % args.k2 == 0:
            if epoch % 5 == 0:
                metrics_dict, images_dict, visuals_dict = eval_vmfnet_mm_sm(model, val_loader, device)

                for key, value in metrics_dict.items():
                    writer.add_scalar(f'Val_metrics/{key}', value, epoch)
                
                for key, value in images_dict.items():
                    writer.add_images(f'Val_images/{key}', value, epoch, dataformats='NCHW')
                
                for key, value in visuals_dict.items():
                    for i in range(args.vc_num):
                        writer.add_images(f'Val_visuals/{key}_{i+1}', value[:,i,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                
                scheduler.step(metrics_dict['DSC_Source'])


                # if best_dice < val_score:
                #     best_dice = val_score
                #     print("Epoch checkpoint")
                #     save_dir = os.path.join(dir_checkpoint, f'fold_{fold}')
                #     os.makedirs(save_dir, exist_ok=True)

                #     print(f"--- Remove old model before saving new one ---")
                #     # Define the pattern to search for existing model files in the directory
                #     existing_model_files = glob.glob(f"{save_dir}/*.pth")
                #     # Delete each found model file
                #     for file_path in existing_model_files:
                #         try:
                #             os.remove(file_path)
                #             print(f"Deleted old model file: {file_path}")
                #         except OSError as e:
                #             print(f"Error deleting file {file_path}: {e}")
                    

                    
                    
                #     torch.save(model.state_dict(), os.path.join(save_dir, f'CP_epoch_{epoch}_dice_{val_score}.pth'))
                #     logging.info('Checkpoint saved !')

            
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
        #dataset_train = MMWHS_single(args, fold_train, labels)
        dataset_train = MMWHS(args, labels, fold_train)
        train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
        print("loading val data")
        #dataset_val = MMWHS_single(args, fold_val, labels) 
        dataset_val = MMWHS(args, labels, fold_val) 
        val_loader = DataLoader(dataset_val, batch_size=args.bs, drop_last=True, num_workers=4)
        len_train_data = dataset_train.__len__()

        train_net(train_loader, val_loader, fold, device, args, num_classes, len_train_data) 
        fold += 1
        break




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)
