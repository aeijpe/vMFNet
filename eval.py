import torch
from tqdm import tqdm
import torch.nn.functional as F
from metrics import dice


def eval_vmfnet(model, loader, device, layer):
    """Evaluation without the densecrf with the dice coefficient"""
    model.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    display_itr = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for imgs, true_masks in loader:
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                rec, pre_seg, features, kernels, L_visuals = model(imgs, layer=layer)

            pred = torch.where(pre_seg > 0.5, 1, 0)
            tot += dice(true_masks, pre_seg, model.num_classes)#.item()

            if display_itr == 5:
                show_imgs = imgs
                show_rec = rec
                show_true = true_masks
                show_pred = pred
            
            display_itr += 1
            pbar.update()
    
    return tot / n_val, show_imgs, show_rec, show_true, show_pred




