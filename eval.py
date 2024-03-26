import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from metrics import dice


from composition.losses import ClusterLoss
from losses import KlDiv


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

            compact_pred_b = torch.argmax(pre_seg, dim=1).unsqueeze(1)
            tot += dice(true_masks, compact_pred_b, model.num_classes)#.item()

            if display_itr == 5:
                show_imgs = imgs
                show_rec = rec
                show_true = true_masks
                show_pred = compact_pred_b
                show_vis = L_visuals
            
            display_itr += 1
            pbar.update()
    
    
    return tot / n_val, show_imgs, show_rec, show_true, show_pred, show_vis


def eval_vmfnet_uns(model, loader, device, layer=8):
    """Evaluation with reconstruction performance"""
    model.eval()
    n_val = len(loader)  # the number of batch
    tot = 0
    display_itr = 0
    l1_distance = nn.L1Loss().to(device)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for imgs, true_masks in loader:
            imgs = imgs.to(device)

            with torch.no_grad():
                rec, features, kernels, L_visuals = model(imgs)

            tot += l1_distance(rec, imgs).item()

            if display_itr == 5:
                show_imgs = imgs
                show_rec = rec
                show_vis = L_visuals
            
            display_itr += 1
            pbar.update()
    
    
    return tot / n_val, show_imgs, show_rec, show_vis



def eval_vmfnet_mm(model, loader, device, layer=8):
    """Evaluation with reconstruction performance"""
    model.eval()
    n_val = len(loader)  # the number of batch
    display_itr = 0
    
    metrics_dict_total = {'Source/reco_loss': 0, 'Source/cluster_loss': 0, 'Source/dice_loss': 0,
                        'Target/reco_loss': 0, 'Target/cluster_loss': 0, 'Target/dice_loss': 0,
                        'kl_loss': 0, 'DSC_Target': 0, 'DSC_Source':0}

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for img_s, label_s, img_t, label_t in loader:
            img_s = img_s.to(device)
            img_t = img_t.to(device)
            label_s = label_s.to(device)
            label_t = label_t.to(device)

            with torch.no_grad():
                metrics_dict, images_dict, visuals_dict  = model.forward_eval(img_s, img_t, label_s, label_t)

            for key, value in metrics_dict.items():
                metrics_dict_total[key] += value

            if display_itr == 5:
                image_dict_show = images_dict
                visual_dict_show = visuals_dict
                
            display_itr += 1
            pbar.update()

    for key, value in metrics_dict.items():
        metrics_dict_total[key] /= n_val
    
    #model.scheduler_content.step(metrics_dict_total['kl_loss'])
    model.scheduler_source.step(metrics_dict_total['Source/reco_loss'] + metrics_dict_total['Source/cluster_loss'] + metrics_dict_total['Source/dice_loss'])
    model.scheduler_target.step(metrics_dict_total['Target/reco_loss'] + metrics_dict_total['Target/cluster_loss'])
    
    return metrics_dict, image_dict_show, visual_dict_show

