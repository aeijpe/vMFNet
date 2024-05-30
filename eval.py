import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils import dice, assd
import numpy as np


from composition.losses import ClusterLoss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 
from losses import KlDiv

from monai.transforms import ScaleIntensity


def eval_vmfnet(model, loader, device, layer=8):
    """Evaluation without the densecrf with the dice coefficient"""
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

def test_vmfnet(model, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
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

            if display_itr == 0:
                show_imgs = imgs
                show_rec = rec
                show_true = true_masks
                show_pred = torch.argmax(pre_seg, dim=1).unsqueeze(1)
                show_vis = L_visuals
            
            dice_batch = dice(true_masks, pre_seg, model.num_classes).cpu().numpy()
            tot += dice_batch
            assd_batch = assd(true_masks, pre_seg, model.num_classes, pix_dim=imgs.meta["pixdim"][1]).cpu()
    
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


def eval_vmfnet_mm(model, loader, device):
    """Evaluation with reconstruction performance"""
    model.eval()
    n_val = len(loader)  # the number of batches
    display_itr = 0

    first = True
    lpips_list_target = []
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device)
    true_assd = 0
    true_dsc = 0
    fake_dsc = 0
    n_val_assd = n_val
    n_val_dsc = n_val
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for img_s, label_s, img_t, label_t in loader:
            img_s = img_s.to(device)
            img_t = img_t.to(device)
            label_s = label_s.to(device)
            label_t = label_t.to(device)

            with torch.no_grad():
                metrics_dict, images_dict, visuals_dict  = model.forward_eval(img_s, img_t, label_s, label_t)
            
            if first:
                metrics_dict_total = metrics_dict
                first = False
            else:
                for key, value in metrics_dict.items():
                    metrics_dict_total[key] += value
                        
            assd_val = metrics_dict["Target/assd"]

            if np.isinf(assd_val).any():
                # print("inf value in assd")
                n_val_assd -= 1
            else:
                true_assd += assd_val
            
            # Skip dice class when no label is there. Background is always there
            if torch.all(label_t==0):
                n_val_dsc -= 1
            else:
                true_dsc += metrics_dict["Target/DSC"]
                fake_dsc += metrics_dict["Target/DSC_fake"]


            img_t3 = torch.cat((img_t, img_t, img_t), dim=1)
            fake_img = ScaleIntensity()(images_dict['Target/fake'])
            fake_img_t3 = torch.cat((fake_img, fake_img, fake_img), dim=1)
            lpips_list_target.append(lpips(fake_img_t3, img_t3))

            if display_itr == 5:
                image_dict_show = images_dict
                visual_dict_show = visuals_dict
                
            display_itr += 1
            pbar.update()

    for key, value in metrics_dict.items():
        metrics_dict_total[key] /= n_val

    if n_val_assd == 0:
        metrics_dict_total["Target/assd"] = 1000
    else:
        metrics_dict_total["Target/assd"] = true_assd / n_val_assd

    metrics_dict_total["Target/DSC"] = true_dsc / n_val_dsc
    metrics_dict_total["Target/DSC_fake"] = fake_dsc / n_val_dsc
    
    return metrics_dict_total, image_dict_show, visual_dict_show, torch.mean(torch.stack(lpips_list_target))


