import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from mmwhs_dataloader import MMWHS_single

import models
from torch.utils.tensorboard import SummaryWriter

from models.unet_model import UNet

import pytorch_lightning as pl

def get_args():
    usage_text = (
        "UNet Pytorch Implementation"
        "Usage:  python pretrain.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-e','--epochs', type= int, default=50, help='Number of epochs')
    parser.add_argument('--seed', default=42, type=int,help='Seed to use for reproducing results') # --> their default was 14
    parser.add_argument('--bs', type= int, default=4, help='Number of inputs per batch')
    parser.add_argument('-c', '--cp', type=str, default='checkpoints', help='The name of the checkpoints.')
    parser.add_argument('--name', type=str, default='test', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-mn','--model_name', type=str, default='unet', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--k1', type=int,  default=40, help='When the learning rate starts decaying')
    parser.add_argument('--k2', type=int,  default=4, help='Check decay learning')
    parser.add_argument('--layer', type=int,  default=8, help='layer from which the deepf eatures are obtained')
    parser.add_argument('--vc_num', type=int,  default=12, help='Kernel/distributions amount')
    parser.add_argument('--data_dir',  type=str, default='../data/other/MR_withGT_proc/annotated/', help='The name of the checkpoints.')

    return parser.parse_args()


def pretrain_fe(args):
    pl.seed_everything(args.seed)

    cp_dir = os.path.join(args.cp, os.path.join(args.name, 'encoder'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = range(0,18)
    

    dataset_train = MMWHS_single(args, cases)
    train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
    n_train = dataset_train.__len__()


    model = UNet(n_classes=1)
    models.initialize_weights(model, args.weight_init)
    model.to(device)
    
    #metrics initialization
    l1_distance = nn.L1Loss().to(device)

    #optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # need to use a more useful lr_scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    log_dir = os.path.join('logs/', args.name)

    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for imgs, true_masks in train_loader:
                imgs = imgs.to(device)
                true_masks = true_masks.to(device)


                out = model(imgs)
                reco = out[0]

                reco_loss = l1_distance(reco, imgs)

                batch_loss = reco_loss
                writer.add_scalar('Loss/reco_loss', reco_loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': batch_loss.item()})

                optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

        if optimizer.param_groups[0]['lr']<=2e-8:
            print('Converge')

        scheduler.step()
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

    # save checkpoint
    print("Epoch checkpoint")
    
    os.makedirs(cp_dir, exist_ok=True)
    logging.info('Created checkpoint directory')
    torch.save(model.state_dict(), os.path.join(cp_dir,'UNet.pth'))
    logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    pretrain_fe(args)