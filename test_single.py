import torch
import os
import argparse
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from eval import eval_vmfnet
from mmwhs_dataloader import MMWHS_single
from models.compcsd import CompCSD
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

import glob

import numpy as np
from metrics import dice



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


def test(model_file, log_dir, data_loader, device, num_classes, cp_dir):
    print("Loading model")
    model = CompCSD(device, 1, args.layer, args.vc_num, num_classes=num_classes, z_length=8, vMF_kappa=30)
    model.initialize(cp_dir)
    model.to(device)
    model.resume(model_file)
    model.eval()


    writer = SummaryWriter(log_dir=log_dir)
    dc_score, imgs, rec, test_true, test_pred, L_visuals = eval_vmfnet(model, data_loader, device)
    
    writer.add_scalar('Dice/test_MRI', dc_score, 0)

    writer.add_images('Test_images/test_MRI', imgs, 0, dataformats='NCHW')
    writer.add_images('Test_images/test_reco_MRI', rec, 0, dataformats='NCHW')
    writer.add_images('Test_images/test_true_MRI', test_true, 0, dataformats='NCHW')
    writer.add_images('Test_images/test_pred_MRI', test_pred, 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_1_MRI', L_visuals[:,0,:,:].unsqueeze(1), 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_2_MRI', L_visuals[:,1,:,:].unsqueeze(1), 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_3_MRI', L_visuals[:,2,:,:].unsqueeze(1), 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_4_MRI', L_visuals[:,3,:,:].unsqueeze(1), 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_5_MRI', L_visuals[:,4,:,:].unsqueeze(1), 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_6_MRI', L_visuals[:,5,:,:].unsqueeze(1), 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_7_MRI', L_visuals[:,6,:,:].unsqueeze(1), 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_8_MRI', L_visuals[:,7,:,:].unsqueeze(1), 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_9_MRI', L_visuals[:,8,:,:].unsqueeze(1), 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_10_MRI', L_visuals[:,9,:,:].unsqueeze(1), 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_11_MRI', L_visuals[:,10,:,:].unsqueeze(1), 0, dataformats='NCHW')
    writer.add_images('L_visuals/L_12_MRI', L_visuals[:,11,:,:].unsqueeze(1), 0, dataformats='NCHW')

    return dc_score
    
    


def main(args):
    pl.seed_everything(args.seed)

    if args.pred == "MYO":
        labels = [1, 0, 0, 0, 0, 0, 0]
        num_classes = 2
    else:
        labels = [1, 2, 3, 4, 5, 6, 7]
        num_classes = 8

    test_fold = [18, 19]

    dataset_test = MMWHS_single(args, test_fold, labels)
    test_loader = DataLoader(dataset_test, batch_size=args.bs, drop_last=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_checkpoints = os.path.join(args.cp, args.name)
    log_dir = os.path.join("logs", args.name)
    pretrained_filenames = glob.glob(os.path.join(dir_checkpoints, "fold_*/*.pth"))
    
    dsc_scores = []
    pretrained_found = False

    for it, model_file in enumerate(pretrained_filenames):
        if it == 2:
            continue
        print(f"Testing model {it+1}/{len(pretrained_filenames)}")
        log_dir_fold = os.path.join(log_dir, f"fold_{it}")
        print("log_dir: ", log_dir_fold)
        dsc = test(model_file, log_dir_fold, test_loader, device, num_classes, dir_checkpoints)
        dsc_scores.append(dsc)
        pretrained_found = True

    if pretrained_found:
        dsc_scores = np.array(dsc_scores)
        mean_dsc = np.mean(dsc_scores)
        std_dsc = np.std(dsc_scores)
        print("FINAL RESULTS")
        print(f"Mean DSC: {mean_dsc}, Std DSC: {std_dsc}")
    else:
        print("No pretrained models found")




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)
