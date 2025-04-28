"""
Main file for training Yolo model on LLVIP dataset.

"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import SimpleYOLO
from dataset import LLVIPDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    plot_loss,
    save_checkpoint,
    load_checkpoint,
)
from loss import SimpleYoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cpu"
BATCH_SIZE = 16 # 64 in original paper..
WEIGHT_DECAY = 0
EPOCHS = 50 # will change it to 1000 when i have a bigger model
NUM_WORKERS = 2
PIN_MEMORY = True
IMG_DIR = "../dataset/LLVIP_small"
RESULTS_DIR = "../results_img"
LOAD_MODEL_FILE = "checkpoint.pth.tar"



def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y, _) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        # print("length of mean_loss list =", len(mean_loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    epoch_loss = sum(mean_loss) / len(mean_loss)
    print(f"Mean loss was {epoch_loss}")
    return epoch_loss


def main():
    losses = []
    LOAD_MODEL = False
    
    # Ask user if they want to load the saved model
    if os.path.exists(LOAD_MODEL_FILE):
        user_input = input(f"Load checkpoint from {LOAD_MODEL_FILE}? (y/n): ").strip().lower()
        if user_input == "y":
            LOAD_MODEL = True
        else:
            LOAD_MODEL = False

    model = SimpleYOLO(split_size=7, num_boxes=2, num_classes=1).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = SimpleYoloLoss()

    # load an existing trained model
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE, weights_only=True), model, optimizer)  # weigts_only=True to not load arbitary code (had a warning about that) !! Pytotch 2.2+!!

    train_dataset = LLVIPDataset(dir_all_images=IMG_DIR)
    
    test_dataset = LLVIPDataset(dir_all_images=IMG_DIR, train=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    for epoch in range(EPOCHS):

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device="cpu"
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        if epoch % 5 == 0:  # every 5 epochs
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            print(f"Saved model at epoch {epoch}")

        losses.append(train_fn(train_loader, model, optimizer, loss_fn))
    
    # plot the loss
    plot_loss(losses, EPOCHS)

    # visualize predicted bounding boxes
    model.eval()

    for x, _, filenames in train_loader: # using train_loader instead of the test_loader to visualize the predictions (bcs the dataset is still small)
        x = x.to(DEVICE)
        with torch.no_grad():
            preds = model(x)

        bboxes_batch = cellboxes_to_boxes(preds)

        for idx in range(x.shape[0]):
            bboxes = non_max_suppression(bboxes_batch[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            
            base_filename = os.path.splitext(filenames[idx])[0]  # removes ".jpg"
            save_path = os.path.join(RESULTS_DIR, f"{base_filename}_prediction.png")

            plot_image(x[idx], bboxes, save_path=save_path)
        break


if __name__ == "__main__":
    main()