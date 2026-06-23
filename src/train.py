import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam

from src.dataset import CocoSegDataset
from src.model import UNet

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

        running_loss += loss.item() * images.size()

    epoch_loss = running_loss / len(loader.dataset)

    return epoch_loss

def main():
    # -------------
    # config
    # -------------
    image_dir = ".../data/train/images"
    json_path = "../data/train/train.json"

    batch_size = 8
    lr = 1e-3
    num_epochs = 20
    val_ratio = 0.2
    random_seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # -------------
    # dataset
    # -------------
    full_dataset = CocoSegDataset(image_dir=image_dir, json_path=json_path)
    # for debugging with one image overfit
    tiny_dataset = torch.utils.data.Subset(full_dataset, [0])

    val_size = int(len(full_dataset)*val_ratio)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size,
    #     shuffle=True,
    #     num_workers=0
    # )
    #
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size,
    #     shuffle=False,
    #     num_workers=0
    # )


    ##### debugging
    train_loader = DataLoader(
        tiny_dataset,
        batch_size=1,
        shuffle=True
    )

    val_loader = DataLoader(
        tiny_dataset,
        batch_size=1,
        shuffle=False
    )

    num_epochs = 200
    ##### end of debugging

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
            torch.save(model.state_dict(), "checkpoint/best_model.pth")
            print("Saved best model")

if __name__ == "__main__":
    main()

