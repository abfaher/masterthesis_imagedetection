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
from loss import SimpleYoloLoss
from utils import (
    mean_average_precision,
    get_bboxes,
    plot_train_val_loss,
    save_checkpoint,
    load_checkpoint,
)

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 1e-3
DEVICE = "cpu"
BATCH_SIZE = 16 # 64 in original paper..
WEIGHT_DECAY = 5e-4
EPOCHS = 300
NUM_WORKERS = 2
PIN_MEMORY = True
IMG_DIR = "../dataset/LLVIP_small"
RESULTS_DIR = "../results_img"
LOAD_MODEL_FILE = "best_model.pth.tar"



def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y, _) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    epoch_loss = sum(mean_loss) / len(mean_loss)
    print(f"Mean loss was {epoch_loss}")
    return epoch_loss


def compute_val_loss(model, val_loader, loss_fn, device="cpu"):
    model.eval()
    val_mean_losses = []
    with torch.no_grad():
        for val_x, val_y, _ in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_out = model(val_x)
            val_loss = loss_fn(val_out, val_y)
            val_mean_losses.append(val_loss.item())
    model.train() # switch back to train mode
    return sum(val_mean_losses) / len(val_mean_losses)



def main():
    losses = []
    val_losses = []
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
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[75, 105],  # à 75 et 105 epochs, on baisse le LR
        gamma=0.1
    )
    loss_fn = SimpleYoloLoss()

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    # load an existing trained model
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE, weights_only=True), model, optimizer)  # weigts_only=True to not load arbitary code (had a warning about that) !! Pytotch 2.2+!!

    # Load full dataset and split into train and val
    full_dataset = LLVIPDataset(dir_all_images=IMG_DIR, train=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    best_val_loss = float('inf')
    patience = 10
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS+1):

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device="cpu"
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        losses.append(train_fn(train_loader, model, optimizer, loss_fn))

        # Validation loss
        avg_val_loss = compute_val_loss(model, val_loader, loss_fn, device=DEVICE)
        val_losses.append(avg_val_loss)
        print(f"Validation loss: {avg_val_loss:.4f}")

        # saving the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filename=LOAD_MODEL_FILE)
            print("Best model saved.")
        
        # Early stopping
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        scheduler.step()
    
    # plot the loss
    plot_train_val_loss(losses, val_losses)


if __name__ == "__main__":
    main()