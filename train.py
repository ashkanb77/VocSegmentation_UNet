import argparse
import logging
from torch import nn
from torch.utils.data import DataLoader
import albumentations as A
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
        A.Resize(530, 530),
        A.RandomCrop(512, 512),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(-20, 20)),
        A.VerticalFlip(p=0.5),
])

test_transforms = A.Compose([
        A.Resize(512, 512)
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


def train(model, train_dataloader, val_dataloader, checkpoint, optimizer, epochs, lr, plot=True):
    criterion = nn.CrossEntropyLoss()

    losses = []
    val_losses = []

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        n_batches = len(train_dataloader)

        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:

            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                images, masks = batch

                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())

        total_loss = total_loss / n_batches

        losses.append(total_loss)

        val_loss = eval(model, val_dataloader, checkpoint)
        val_losses.append(val_loss)

        print(
            f"Epoch: {epoch + 1}, Train Loss: {total_loss:.4}," \
            + f" Val Loss: {val_loss: .4}"
        )

    if plot:
        plt.title('Loss')
        plt.plot(losses, label='Train Loss')
        plt.plot(val_losses, label='Test Loss')
        plt.legend(loc='best')
        plt.show()

    return losses, val_losses


def eval(model, val_dataloader, checkpoint):
    criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0
    n_batches = len(val_dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            images, masks = batch

            outputs = model(images)

            loss = criterion(outputs, masks)
            total_loss += loss.item()

    total_loss = total_loss / n_batches

    checkpoint.save(total_loss)

    return total_loss


model = UNet(classes=22).to(device)
optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
checkpoint = Checkpoint(model, args.model_name, args.checkpoint_dir)

train(
    model, train_dataloader, val_dataloader, checkpoint, optimizer, args.n_epochs, args.learning_rate
    )
