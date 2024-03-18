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
    l1 = 0
    clu = 0
    dsc = 0
    display_itr = 0
    l1_distance = nn.L1Loss().to(device)
    clu_loss = ClusterLoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for img_s, label_s, img_t, label_t in loader:
            img_s = img_s.to(device)
            img_t = img_t.to(device)
            label_t = label_t.to(device)

            with torch.no_grad():
                pre_seg, rec, features_t, kernels, norm_vmf_activations_t, norm_vmf_activations_s, features_s  = model.forward_eval(img_s, img_t)

            l1 += l1_distance(rec, img_t).item()
            clu += clu_loss(features_t.detach(), kernels).item()

            compact_pred_b = torch.argmax(pre_seg, dim=1).unsqueeze(1)
            dsc += dice(label_t, compact_pred_b, model.num_classes)

            if display_itr == 5:
                show_imgs = img_t
                show_rec = rec
                show_vis_target = norm_vmf_activations_t
                show_vis_source = norm_vmf_activations_s
                true_label = label_t
                pred_label = compact_pred_b
            
            display_itr += 1
            pbar.update()
    
    
    return l1 / n_val, clu / n_val, dsc / n_val , show_imgs, show_rec, true_label, pred_label, show_vis_source, show_vis_target



