from monai.metrics import DiceMetric
import torch
import torch.nn.functional as F


def dice(labels, compact_pred, n_class):
    # Initialize the DiceMetric object
    # set reduction to 'none' to get the score for each class separately
    dice_metric = DiceMetric(reduction="mean")
    # print(compact_pred.shape)
    # print(labels.shape)

    compact_pred_oh = F.one_hot(compact_pred.long().squeeze(1), n_class).permute(0, 3, 1, 2)
    labels_oh = F.one_hot(labels.long().squeeze(1), n_class).permute(0, 3, 1, 2)
    # print(labels_oh.shape)

    # Compute the Dice score
    dice_metric(y_pred=compact_pred_oh, y=labels_oh)
    metric = dice_metric.aggregate().item()
    dice_metric.reset()

    return metric