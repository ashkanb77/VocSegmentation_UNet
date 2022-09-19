import argparse
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
import albumentations as A
from torchmetrics import JaccardIndex
import torchvision
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config import *
from utils import *
from dataset import VOCSeg
import pickle
import json


logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger('')

parser = argparse.ArgumentParser()

parser.add_argument('--n_epochs', type=int, default=EPOCHS, help='number of epochs for training')
parser.add_argument('--image_size', type=int, default=IMAGE_SIZE, help='number of epochs for training')
parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR, help='dataset directory')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='learning rate')
parser.add_argument('--experiment', type=str, default='experiment1', help='experiment path')
parser.add_argument('--checkpoint_dir', type=str, default=CHECK_DIR, help='dataset directory')
parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='dataset directory')

args = parser.parse_args()


train_transforms = A.Compose([
        A.Resize(args.image_size + 30, args.image_size + 30),
        A.RandomCrop(args.image_size, args.image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(-20, 20)),
        A.VerticalFlip(p=0.5),
])

test_transforms = A.Compose([
        A.Resize(args.image_size, args.image_size)
])

train_dataset = VOCSeg(
    'dataset', download=True, image_set='train',
     transforms=train_transforms,
     )

val_dataset = VOCSeg(
    'dataset', download=False, image_set='val',
    transforms=test_transforms
     )

train_dataloader = DataLoader(train_dataset, args.batch_size, True)
val_dataloader = DataLoader(val_dataset, args.batch_size, True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, train_dataloader, val_dataloader, checkpoint, optimizer, epochs, lr, plot=True):
    criterion = nn.CrossEntropyLoss()
    jaccard = JaccardIndex(num_classes=22)

    losses = []
    val_losses = []
    mioues = []
    val_mioues = []

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        miou = 0
        n_batches = len(train_dataloader)

        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:

            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                images, masks = batch
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(outputs, masks.long())
                iou = jaccard(outputs, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                miou += iou

                tepoch.set_postfix(loss=loss.item(), MIOU=iou)

        total_loss = total_loss / n_batches
        miou = miou / n_batches

        losses.append(total_loss)
        mioues.append(miou)

        val_loss, val_miou = eval(model, val_dataloader, checkpoint)
        val_losses.append(val_loss)
        val_mioues.append(val_miou)

        print(
            f"Epoch: {epoch + 1}, Train Loss: {total_loss:.4}, Train MIOU: {miou:.4}" \
            + f" Val Loss: {val_loss: .4}, Val MIOU: {val_miou:.4}"
        )

    if plot:
        plt.title('Loss')
        plt.plot(losses, label='Train Loss')
        plt.plot(val_losses, label='Test Loss')
        plt.legend(loc='best')
        plt.show()

        plt.title('MIOU')
        plt.plot(mioues, label='Train MIOU')
        plt.plot(val_mioues, label='Test MIOU')
        plt.legend(loc='best')
        plt.show()

    return losses, val_losses, mioues, val_mioues


def eval(model, val_dataloader, checkpoint):
    criterion = nn.CrossEntropyLoss()
    jaccard = JaccardIndex(num_classes=22)

    model.eval()

    total_loss = 0
    miou = 0
    n_batches = len(val_dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks.long())
            iou = jaccard(outputs, masks)
            total_loss += loss.item()
            miou += iou

    total_loss = total_loss / n_batches
    miou = miou / n_batches

    checkpoint.save(total_loss)

    return total_loss, miou


model = UNet(classes=22).to(device)
optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
checkpoint = Checkpoint(model, args.model_name, args.checkpoint_dir)

train(
    model, train_dataloader, val_dataloader, checkpoint, optimizer, args.n_epochs, args.learning_rate
    )
