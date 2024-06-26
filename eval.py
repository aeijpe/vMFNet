import torch
from tqdm import tqdm
from utils import dice, assd
import numpy as np


# Validate the model (in train.py)
def eval_vmfnet(model, loader, device, layer=8):
    model.eval()
    n_val = len(loader)  # the number of batch
    tot = 0
    display_itr = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for imgs, true_masks in loader:
            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            with torch.no_grad():
                rec, pre_seg, features, kernels, L_visuals = model(imgs)

            # Randomly choose some images to visualize in tensorboard
            if display_itr == 0:
                tot = dice(true_masks, pre_seg, model.num_classes)
                show_imgs = imgs
                show_rec = rec
                show_true = true_masks
                show_pred = torch.argmax(pre_seg, dim=1).unsqueeze(1)
                show_vis = L_visuals
            else:
                tot += dice(true_masks, pre_seg, model.num_classes)
           
            display_itr += 1
            pbar.update()
    
    
    return tot / n_val, show_imgs, show_rec, show_true, show_pred, show_vis


# Test the model (in test.py)
def test_vmfnet(model, loader, device):
    model.eval()
    n_val = len(loader)  # the number of batch
    tot = np.zeros(model.num_classes)
    tot_assd = 0
    tot_dsc = 0
    n_assd = n_val
    n_val_dsc = n_val
    display_itr = 0
    with tqdm(total=n_val, desc='Test round', unit='batch', leave=False) as pbar:
        for imgs, true_masks in loader:
            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            with torch.no_grad():
                rec, pre_seg, features, kernels, L_visuals = model(imgs)

            # Randomly choose some images to visualize in tensorboard
            if display_itr == 0:
                show_imgs = imgs
                show_rec = rec
                show_true = true_masks
                show_pred = torch.argmax(pre_seg, dim=1).unsqueeze(1)
                show_vis = L_visuals
            
            # calculate metrics
            dice_batch = dice(true_masks, pre_seg, model.num_classes).cpu().numpy()
            tot += dice_batch
            assd_batch = assd(true_masks, pre_seg, model.num_classes, pix_dim=imgs.meta["pixdim"][1]).cpu()
    
            # if label or prediction is all 0
            if np.isinf(assd_batch).any():
                n_assd -= 1
            else:
                tot_assd += assd_batch.item()
            
            if torch.all(true_masks==0):
                n_val_dsc -= 1
            else:
                tot_dsc += dice_batch[1]
                
            display_itr += 1
            pbar.update()
    
    
    return tot / n_val, tot_assd / n_assd, tot_dsc/n_val_dsc, show_imgs, show_rec, show_true, show_pred, show_vis