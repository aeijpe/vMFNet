import torch
import os
import argparse
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from eval import eval_vmfnet_mm
from mmwhs_dataloader import MMWHS

from models.crosscompcsd import CrossCSD
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
    #testing details
    parser.add_argument('--seed', default=42, type=int,help='Seed to use for reproducing results') # --> their default was 14
    parser.add_argument('--bs', type= int, default=4, help='Number of inputs per batch')
    parser.add_argument('--cp', type=str, default='checkpoints/', help='The name of the checkpoints.')

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
    parser.add_argument('--init', type=str, default='pretrain', help='Initialization method') # pretrain (original), xavier, cross.
    
    return parser.parse_args()


def test(model_file, log_dir, data_loader, device, num_classes, cp_dir):
    print("Loading model")
    model = CrossCSD(args, device, 1, num_classes, vMF_kappa=30)
    model.to(device)
    model.resume(model_file)
    model.eval()

    writer = SummaryWriter(log_dir=log_dir)
    metrics_dict, images_dict, visuals_dict = eval_vmfnet_mm(model, data_loader, device)

    for key, value in metrics_dict.items():
        writer.add_scalar(f'Test_metrics/{key}', value, 0)
    
    for key, value in images_dict.items():
        writer.add_images(f'Test_images/{key}', value, 0, dataformats='NCHW')
    
    for key, value in visuals_dict.items():
        for i in range(args.vc_num):
            writer.add_images(f'Test_visuals/{key}_{i+1}', value[:,i,:,:].unsqueeze(1), 0, dataformats='NCHW')

    return metrics_dict["DSC_Target"]
    
    


def main(args):
    pl.seed_everything(args.seed)

    if args.pred == "MYO":
        labels = [1, 0, 0, 0, 0, 0, 0]
        num_classes = 2
    else:
        labels = [1, 2, 3, 4, 5, 6, 7]
        num_classes = 8

    test_fold = [18, 19]

    dataset_test = MMWHS(args, labels, test_fold)
    test_loader = DataLoader(dataset_test, batch_size=args.bs, drop_last=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_checkpoints = os.path.join(args.cp, args.name)
    log_dir = os.path.join("logs", args.name)
    pretrained_filenames = glob.glob(os.path.join(dir_checkpoints, "fold_*/*.pth"))
    
    dsc_scores = []
    pretrained_found = False

    for it, model_file in enumerate(pretrained_filenames):
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
