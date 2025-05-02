import os
import torch
from model import SimpleYOLO
from utils import load_checkpoint, cellboxes_to_boxes, non_max_suppression, plot_image
from dataset import LLVIPDataset
from torch.utils.data import DataLoader

DEVICE = "cpu"
LOAD_MODEL_FILE = "checkpoint.pth.tar"  # adapte selon ton fichier
IMG_DIR = "../dataset/LLVIP_small"
save_dir = "../results_testloader"
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True

def visualize_predictions(model, loader, save_dir, device=DEVICE):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for x, _, filenames in loader:
            x = x.to(device)
            preds = model(x)
            bboxes_batch = cellboxes_to_boxes(preds)

            for idx in range(x.shape[0]):
                bboxes = non_max_suppression(
                    bboxes_batch[idx], iou_threshold=0.5, threshold=0.6, box_format="midpoint"
                )
                base_filename = os.path.splitext(filenames[idx])[0]
                save_path = os.path.join(save_dir, f"{base_filename}_test_prediction.png")
                plot_image(x[idx], bboxes, save_path=save_path)



if __name__ == "__main__":
    model = SimpleYOLO(split_size=7, num_boxes=2, num_classes=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # pas super important ici

    checkpoint = torch.load(LOAD_MODEL_FILE, map_location=DEVICE)
    load_checkpoint(checkpoint, model, optimizer)

    test_dataset = LLVIPDataset(dir_all_images=IMG_DIR, train=False)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    visualize_predictions(model, test_loader, save_dir, DEVICE)
