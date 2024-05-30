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








def pretrain(args, labels, device, dataset_type):
    cases = range(0,18)
    dataset_train = dataset_type(args, labels, cases)
    train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
    len_data = dataset_train.__len__()

    dir_checkpoint = os.path.join(args.cp, args.name)
    save_dir = os.path.join(dir_checkpoint, f'pretrain')
    os.makedirs(save_dir, exist_ok=True)

    #Model selection and initialization
    model = CrossCSDFirst(args, device, 1)
    model.to(device)

    log_dir = os.path.join('logs', os.path.join(args.name, 'pretraining'))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_score = np.inf

    global_step = 0

    show_img_s, _, show_img_t, _ = next(iter(train_loader)) 
    show_img_s = show_img_s.to(device)
    show_img_t = show_img_t.to(device)  

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
    
            
            if epoch % 10 == 0:
                images_dict = model.forward_eval(show_img_s, show_img_t)
                for key, value in images_dict.items():
                        writer.add_images(f'Val_images/{key}', value, epoch, dataformats='NCHW')

            if epoch % 5 == 0:
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