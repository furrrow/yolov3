import torch
import torch.nn as nn
from utils import intersection_over_union
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/loss.py


class YoloLoss(nn.Module):
    """
    loss for the yolo v1 model
    s is the split size of image, (paper has 7)
    b is the number of boxes (paper has 2)
    c is the number of classes
    """
    def __init__(self, split=7, n_box=2, n_classes=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.s = split
        self.b = n_box
        self.c = n_classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # note [..., 0] and [:, :, :, 0] are the same in 4d list.
        # predictions are shaped (BATCH_SIZE, s*s(c+b*5))
        predictions = predictions.reshape(-1, self.s, self.s, self.c + self.b * 5)
        # TODO: figure out why the indicies are the way they are
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)  # best_box here is basically argmax
        exists_box = target[..., 20].unsqueeze(3)  # identity of obj_i

        """box coordinates loss """
        box_predictions = exists_box * (
            (
                    best_box * predictions[..., 26:30]
                    + (1 - best_box) * predictions[..., 21:25]
            )
        )
        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) \
                                    * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        # (N, S, S, 4) -> (N * S * S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )
        """object loss"""
        pred_box = (
            best_box * predictions[..., 25:26] + (1-best_box) * predictions[..., 20:21]
        )
        # (N*S*S, 1)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )
        """no object loss"""
        # (N, S, S, 1) -> (N, S*S*1)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        """class loss"""
        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )
        """ actual loss """
        loss = (
            self.lambda_coord * box_loss  # first two rows of loss in paper
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss

