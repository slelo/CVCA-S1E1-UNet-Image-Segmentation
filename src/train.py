import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam

from src.dataset import CocoSegDataset
from src.model import UNet

from pathlib import Path

import argparse

path = "../checkpoints"

#CONTROLS
#set the range of the images here (0 means all image should be used for training)
train_range = 0
batch_size = 8
num_epochs = 200
use_cuda = True

def compute_iou(logits, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.item()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        #forward
        logits = model(images)

        loss = criterion(logits, masks)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = running_loss / len(loader.dataset)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss =criterion(logits, masks)

        iou = compute_iou(logits, masks)

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)

    return epoch_loss

def main(args):
    train_range = args.train_range
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    print("Running training with "
          "train_range=", train_range,
          ", batch_size=", batch_size,
          ", num_epochs=", num_epochs,".")

    # -------------
    # config
    # -------------
    image_dir = "../data/train/images"
    json_path = "../data/train/train.json"

    lr = 1e-3
    val_ratio = 0.2
    random_seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("Using device: ", device)

    # -------------
    # dataset
    # -------------
    full_dataset = CocoSegDataset(image_dir=image_dir, json_path=json_path)
    # taking a subset or full dataset for training
    tiny_dataset = full_dataset \
       if train_range==0 else \
       (torch.utils.data.Subset(full_dataset, list(range(train_range))))

    # tiny_dataset = torch.utils.data.Subset(full_dataset, [0])

    val_size = int(len(full_dataset)*val_ratio)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        tiny_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        tiny_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    # -------------
    # model / loss / optimizer
    # -------------
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer=Adam(model.parameters(), lr=lr)

    # -------------
    # training loop
    # -------------
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f}"
        )

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path+"/best_model.pth")
            print("Saved best model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_range", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--use_cuda", type=bool, default=True)

    args = parser.parse_args()
    main(args)

