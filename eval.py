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
    
    return metrics_dict_total, image_dict_show, visual_dict_show


def eval_vmfnet_mm_sm(model, loader, device, layer=8):
    """Evaluation with reconstruction performance"""
    model.eval()
    n_val = len(loader)  # the number of batch
    display_itr = 0
    l1_distance = nn.L1Loss().to(device)
    cluster_loss = ClusterLoss()
    
    metrics_dict = {'Source/reco_loss': 0, 'Source/cluster_loss': 0,
                        'Target/reco_loss': 0, 'Target/cluster_loss': 0,
                         'DSC_Target': 0, 'DSC_Source':0} 

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for img_s, label_s, img_t, label_t in loader:
            img_s = img_s.to(device)
            img_t = img_t.to(device)
            label_s = label_s.to(device)
            label_t = label_t.to(device)

            with torch.no_grad():
                rec_s, pre_seg_s, features_s, kernels_s, L_visuals_s = model(img_s)
                rec_t, pre_seg_t, features_t, kernels_t, L_visuals_t = model(img_t)

        

            gt_oh_s = F.one_hot(label_s.long().squeeze(1), num_classes=model.num_classes).permute(0, 3, 1, 2)
            reco_loss_s = l1_distance(rec_s, img_s)
            clu_loss_s = cluster_loss(features_s.detach(), kernels_s) 

            compact_pred_s = torch.argmax(pre_seg_s, dim=1).unsqueeze(1)
            dsc_source = dice(label_s, compact_pred_s, model.num_classes)


            gt_oh_t = F.one_hot(label_t.long().squeeze(1), num_classes=model.num_classes).permute(0, 3, 1, 2)
            reco_loss_t = l1_distance(rec_t, img_t)
            clu_loss_t = cluster_loss(features_t.detach(), kernels_t)

            compact_pred_t = torch.argmax(pre_seg_t, dim=1).unsqueeze(1)
            dsc_target = dice(label_t, compact_pred_t, model.num_classes)

            if display_itr == 5:
                images_dict = {'Source_image': img_s, 'Target_image': img_t, 'Reconstructed_source': rec_s, 'Reconstructed_target': rec_t, 
                       'Segmentation_source': compact_pred_s, 'Segmentation_target': compact_pred_t, 'Source_label': label_s, 'Target_label': label_t}
        
                visuals_dict = {'Source': L_visuals_s, 'Target': L_visuals_t}


            metrics_dict['Source/reco_loss'] += reco_loss_s.item()
            metrics_dict['Source/cluster_loss'] += clu_loss_s.item()
            metrics_dict['Target/reco_loss'] += reco_loss_t.item()
            metrics_dict['Target/cluster_loss'] += clu_loss_t.item()
            metrics_dict['DSC_Target'] += dsc_target
            metrics_dict['DSC_Source'] += dsc_source

            display_itr += 1
            pbar.update()

    for key, value in metrics_dict.items():
        metrics_dict[key] /= n_val
    

    return metrics_dict, images_dict, visuals_dict
