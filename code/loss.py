import torch
import torch.nn as nn
from utils import intersection_over_union


class SimpleYoloLoss(nn.Module):
    """
    Compute the loss for one layer yolo (v1) model based on LLVIP Dataset.
    """

    def __init__(self, S=7, B=2, C=1):
        super(SimpleYoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (using LLVIP dataset is 1),
        """
        self.S = S
        self.B = B
        self.C = C

        # From Yolo paper: how much we should pay loss for no object (noobj) 
        # and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        # and we need to reshape them to (BATCH_SIZE, S, S, C+B*5)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., 2:6], target[..., 2:6])  # FIRST BOX
        iou_b2 = intersection_over_union(predictions[..., 7:11], target[..., 2:6]) # SECOND BOX
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        _, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 1].unsqueeze(3)  # in paper this is Iobj_i (1 instead of 20 ??)


        #  -------- FOR BOX COORDINATES --------   #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 7:11]
                + (1 - bestbox) * predictions[..., 2:6]
            )
        )

        box_targets = exists_box * target[..., 2:6]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),  # , should be removed ?
        )


        #  -------- FOR OBJECT LOSS --------   #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 6:7] + (1 - bestbox) * predictions[..., 1:2]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 1:2]),
        )

        #  -------- FOR NO OBJECT LOSS --------   #
        # we take the loss for both boxes (both of them should know that there's no object)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 1:2], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 1:2], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 6:7], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 1:2], start_dim=1)
        )

        #  -------- FOR CLASS LOSS --------  #

        # (N, S, S, 1) -> (N*S*S, 1)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :1], end_dim=-2,),
            torch.flatten(exists_box * target[..., :1], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss