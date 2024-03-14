import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from eval import eval_vmfnet_uns
from mmwhs_dataloader import MMWHS_single
from models.compcsd_rec import CompCSDRec
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
    parser.add_argument('-e','--epochs', type= int, default=100, help='Number of epochs')
    parser.add_argument('--seed', default=42, type=int,help='Seed to use for reproducing results') # --> their default was 14
    parser.add_argument('--bs', type= int, default=4, help='Number of inputs per batch')
    parser.add_argument('-c', '--cp', type=str, default='checkpoints', help='The name of the checkpoints.')
    parser.add_argument('-n','--name', type=str, default='test', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-mn','--model_name', type=str, default='compcsd', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--k1', type=int,  default=40, help='When the learning rate starts decaying')
    parser.add_argument('--k2', type=int,  default=4, help='Check decay learning')
    parser.add_argument('--layer', type=int,  default=8, help='layer from which the deepf eatures are obtained')
    parser.add_argument('--vc_num', type=int,  default=12, help='Kernel/distributions amount')
    parser.add_argument('--data_dir',  type=str, default='../data/other/MR_withGT_proc/annotated/', help='The name of the checkpoints.')
    parser.add_argument('--pred', type=str, default='MYO', help='Segmentation task')
    parser.add_argument('--k_folds', type= int, default=5, help='Cross validation')



    return parser.parse_args()


def train_net(train_loader, val_loader, fold, device, args, len_train_data):
    best_dice = 0
    dir_checkpoint = os.path.join(args.cp, args.name)

    #Model selection and initialization
    model = CompCSDRec(device, 1, args.layer, args.vc_num, vMF_kappa=30)
    # num_params = utils.count_parameters(model)
    # print('Model Parameters: ', num_params)
    model.initialize(dir_checkpoint, args.weight_init)
    model.to(device)

    #metrics initialization
    l1_distance = nn.L1Loss().to(device)
    cluster_loss = ClusterLoss()

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
            for imgs, true_masks in train_loader:
                imgs = imgs.to(device)

                rec, features, kernels, L_visuals = model(imgs)


                reco_loss = l1_distance(rec, imgs)
                clu_loss = cluster_loss(features.detach(), kernels)

                batch_loss = reco_loss + clu_loss 

                pbar.set_postfix(**{'loss (batch)': batch_loss.item()})

                optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                writer.add_scalar('loss/batch_loss', batch_loss.item(), global_step)
                writer.add_scalar('loss/reco_loss', reco_loss.item(), global_step)
                writer.add_scalar('loss/cluster_loss', clu_loss.item(), global_step)

                pbar.update(imgs.shape[0])

                global_step += 1

            if optimizer.param_groups[0]['lr'] <= 2e-8:
                print('Converge')

            #if (epoch + 1) > args.k1 and (epoch + 1) % args.k2 == 0:
            if epoch % 5 == 0:
                val_score, imgs, rec, test_true, test_pred, L_visuals = eval_vmfnet_uns(model, val_loader, device)
             
                scheduler.step(val_score)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                logging.info('Validation Dice Coeff: {}'.format(val_score))

                writer.add_scalar('Dice/val', val_score, epoch)

                writer.add_images('Val_images/test', imgs, epoch, dataformats='NCHW')
                writer.add_images('Val_images/test_reco', rec, epoch, dataformats='NCHW')
                writer.add_images('Val_images/test_true', test_true, epoch, dataformats='NCHW')
                writer.add_images('Val_images/test_pred', test_pred, epoch, dataformats='NCHW')

                writer.add_images('L_visuals/L_1', L_visuals[:,0,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals/L_2', L_visuals[:,1,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals/L_3', L_visuals[:,2,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals/L_4', L_visuals[:,3,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals/L_5', L_visuals[:,4,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals/L_6', L_visuals[:,5,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals/L_7', L_visuals[:,6,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals/L_8', L_visuals[:,7,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals/L_9', L_visuals[:,8,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals/L_10', L_visuals[:,9,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals/L_11', L_visuals[:,10,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                writer.add_images('L_visuals/L_12', L_visuals[:,11,:,:].unsqueeze(1), epoch, dataformats='NCHW')


                if best_dice < val_score:
                    best_dice = val_score
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
                    
                    torch.save(model.state_dict(), os.path.join(save_dir, f'CP_epoch_{epoch}_dice_{val_score}.pth'))
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
        if fold == 0:
            fold += 1
            continue
        print('Train fold:', fold_train)
        print('Val fold:', fold_val)
        print("loading train data")
        dataset_train = MMWHS_single(args, fold_train, labels)
        train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
        print("loading val data")
        dataset_val = MMWHS_single(args, fold_val, labels) 
        val_loader = DataLoader(dataset_val, batch_size=args.bs, drop_last=True, num_workers=4)
        len_train_data = dataset_train.__len__()

        train_net(train_loader, val_loader, fold, device, args, num_classes, len_train_data) 
        fold += 1




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)
