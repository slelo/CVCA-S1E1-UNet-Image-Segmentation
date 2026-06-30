import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam

from src.dataset import CocoSegDataset
from src.model import UNet

from pathlib import Path

import argparse

path = "../checkpoints"




def compute_iou(logits, masks, threshold=0.5, eps=1e-6):
    """
    Intersection over Union metric. Very useful for labeling tasks
    Learn more: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    :param logits: raw logits of the model from the output layer
    :param masks: array of ground truth masks
    :param threshold: the lower limit for a logit value to consider the pixel as a vehicle
    :param eps: small constant value to avoid zero-division
    :return:
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    masks = (masks > 0).float()

    assert preds.shape == masks.shape, f"Shape mismatch: preds {preds.shape}, masks {masks.shape}"

    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection

    iou = (intersection + eps) / (union + eps)

    return iou.item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    # Training one epoch based on the defined parameters.
    :param model: The model we built
    :param loader: Dataloader of split dataset for training (see main())
    :param optimizer: The optimzer we use to adjust the weights and learning rate of neural network.
    In this project we use Adam.
    :param criterion: Our loss function (Criterion is a PyTorch abstraction to standardize loss functions,
    an instance of torch.nn. See more: https://docs.pytorch.org/docs/2.12/nn.html#loss-functions)
    :param device: The device we use to run the training. In our case CUDA or CPU
    :return: epoch_loss and epoch_iou
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        #forward propagation
        logits = model(images)
        loss = criterion(logits, masks)

        iou = compute_iou(logits, masks)

        train_batch_size = images.size(0)

        #backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #calculating the loss
        running_loss += loss.item() * train_batch_size
        running_iou += iou * train_batch_size


    epoch_loss = running_loss / len(loader.dataset)
    epoch_iou = running_iou / len(loader.dataset)

    return epoch_loss, epoch_iou

@torch.no_grad() # this decorator tells PyTorch not to keep track of the gradients. We only need that for training.
def validate_one_epoch(model, loader, criterion, device):
    """
    Validating model training.
    :param model: The model we built
    :param loader: Dataloader of split dataset for training (see main())
    :param criterion: Our loss function (Criterion is a PyTorch abstraction to standardize loss functions,
    an instance of torch.nn. See more: https://docs.pytorch.org/docs/2.12/nn.html#loss-functions)
    :param device: The device we use to run the training. In our case CUDA or CPU
    :return: epoch_loss and epoch_iou
    :return:
    """
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss =criterion(logits, masks)

        val_batch_size = images.size(0)
        iou = compute_iou(logits, masks)

        # calculating the loss
        running_loss += loss.item() * val_batch_size
        running_iou += iou * val_batch_size

    epoch_loss = running_loss / len(loader.dataset)
    epoch_iou = running_iou / len(loader.dataset)

    return epoch_loss, epoch_iou

def main(args):
    train_range = args.train_range
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    use_cuda = args.use_cuda
    print("Value of use_cuda is now ", use_cuda)

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

    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
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

    #generates random numbers
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
    best_val_iou = float("inf")

    for epoch in range(num_epochs):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"train_iou: {train_iou:.4f} | "
            f"val_iou: {val_iou:.4f}"
        )

        # save the model with best loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path+"/best_loss_model.pth")
            print("Saved best LOSS model")

        # save the model with best iou
        if val_iou < best_val_iou:
            best_val_iou = val_iou
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path + "/best_iou_model.pth")
            print("Saved best IoU model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    def str_to_bool(value):
        if value.lower() in ("true", "1", "yes"):
            return True
        elif value.lower() in ("false", "0", "no"):
            return False
        else:
            raise ValueError("Expected true/false")

    #Default values:

    # train_range
    # train_range sets the range of the images to
    # train the model (0 means all image should be used for training)


    # batch_size
    # batch_size hyperparameter controls the number of training examples
    # processed by a model before the model's weights are updates

    # num_epochs
    # num_epochs hyperparameter controls how many training cycles we
    # want to have over the range of train_range

    # use_cuda
    # CUDA should be used if possible to accelerate training. Change it to False in case
    # of unexpected errors or for debugging
    parser.add_argument("--train_range", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--use_cuda", type=str_to_bool, default=True)

    args = parser.parse_args()
    main(args)

