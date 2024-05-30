import torch
import os
import argparse
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from eval import test_vmfnet
from mmwhs_dataloader import MMWHS_single
from chaos_dataloader import CHAOS_single
from models.compcsd import CompCSD
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

import glob

import numpy as np
from utils import *



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
    parser.add_argument('--bs', type= int, default=1, help='Number of inputs per batch')
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
    parser.add_argument('--init', type=str, default='xavier', help='Initialization method') # pretrain (original), xavier, cross.
    parser.add_argument('--data_type', default='MMWHS', type=str, help='Baseline used') 

    return parser.parse_args()


def test(save_dir, data_loader, writer, device, num_classes, fold):
    pretrained_model = glob.glob(os.path.join(save_dir, "*.pth"))

    if pretrained_model == []:
        print("no pretrained model found!")
        quit()
    else:
        model_file = pretrained_model[0]
    print("Loading model")
    print(model_file)

    model = CompCSD(device, 1, args.layer, args.vc_num, num_classes=num_classes, z_length=8, vMF_kappa=30, init=args.init)
    model.initialize(save_dir, init="xavier") # Does not matter for testing
    model.to(device)
    model.resume(model_file)
    model.eval()

    dsc_classes, assd, true_dsc, imgs, rec, test_true, test_pred, L_visuals = test_vmfnet(model, data_loader, device)
    
    for i, item in enumerate(dsc_classes):
            writer.add_scalar(f'Test_metrics/dice_class_{i}', item, 0)

    writer.add_images(f'Test_images/image', imgs, 0, dataformats='NCHW')
    writer.add_images(f'Test_images/image_reco', rec, 0, dataformats='NCHW')
    writer.add_images(f'Test_images/mask_true', test_true, 0, dataformats='NCHW')
    writer.add_images(f'Test_images/mask_pred', test_pred, 0, dataformats='NCHW')

    for i in range(10):
        writer.add_images(f'L_visuals/L_{i}', L_visuals[:,i,:,:].unsqueeze(1), 0, dataformats='NCHW')

    return dsc_classes[0], dsc_classes[1], assd, true_dsc
    
def test_k_folds(args, labels, num_classes, device, dataset_type):
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0
    dsc_scores_BG = []
    dsc_scores = []
    assd_scores = []
    true_dsc_scores = []

    print("TASK: ", args.pred)

    # for fold_train, fold_test_val in kf.split(cases):
    for fold_train_val, fold_test in kf.split(cases):
        print("fold")
        dir_checkpoint = os.path.join(args.cp, args.name)
        save_dir = os.path.join(dir_checkpoint, f'fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)
        log_dir = os.path.join('logs', os.path.join(args.name, 'fold_{}'.format(fold)))
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    
        print("loading test data")
        dataset_test = dataset_type(args.data_dir, fold_test, labels) 
        test_loader = DataLoader(dataset_test, batch_size=1, num_workers=4)
        dsc_0, dsc_1, assd, true_dsc = test(save_dir, test_loader, writer, device, num_classes, fold)
        fold += 1

        dsc_scores.append(dsc_1)
        dsc_scores_BG.append(dsc_0)
        assd_scores.append(assd)
        true_dsc_scores.append(true_dsc)

    dsc_scores_BG = np.array(dsc_scores_BG)
    mean_dsc_BG = np.mean(dsc_scores_BG)
    std_dsc_BG = np.std(dsc_scores_BG)
    print("FINAL RESULTS BG")
    print("DSC_0: ", dsc_scores_BG)
    print(f"Mean DSC_0: {mean_dsc_BG}, Std DSC_0: {std_dsc_BG}")

    dsc_scores = np.array(dsc_scores)
    mean_dsc = np.mean(dsc_scores)
    std_dsc = np.std(dsc_scores)
    print("FINAL RESULTS DSC")
    print("DSC_1: ", dsc_scores)
    print(f"Mean DSC_1: {mean_dsc}, Std DSC_1: {std_dsc}")

    true_dsc_scores = np.array(true_dsc_scores)
    mean_true_dsc = np.mean(true_dsc_scores)
    std_true_dsc = np.std(true_dsc_scores)
    print("FINAL RESULTS TRUE DSC")
    print("DSC_1: ", true_dsc_scores)
    print(f"Mean DSC_1: {mean_true_dsc}, Std DSC_1: {std_true_dsc}")

    assd_scores = np.array(assd_scores)
    mean_assd = np.mean(assd_scores)
    std_assd = np.std(assd_scores)
    print("ASSD: ", assd_scores)
    print(f"Mean ASSD: {mean_assd}, Std ASSD: {std_assd}")


def main(args):
    set_seed(args.seed)

    labels, num_classes = get_labels(args.pred)
    # MMWHS Dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS_single
    elif args.data_type == "chaos":
        dataset_type = CHAOS_single
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    test_k_folds(args, labels, num_classes, device, dataset_type)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)
