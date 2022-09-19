import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from config import device
from model import UNet


PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
    + [[0, 0, 0] for i in range(256 - 22)]
    + [[255, 255, 255]],
    dtype=np.uint8,
)


class Checkpoint:

    def __init__(self, model, file_name, dir_path):
        self.best_loss = 1000
        self.folder = dir_path
        self.model = model
        self.file_name = file_name
        os.makedirs(self.folder, exist_ok=True)

    def save(self, loss):

        if loss < self.best_loss:
            state = {
                'model': self.model.state_dict(),
                'loss': loss
            }
            path = os.path.join(os.path.abspath(self.folder), self.file_name + '.pth')
            torch.save(state, path)
            self.best_loss = loss

    @staticmethod
    def load_model(path):
        checkpoint = torch.load(path, map_location=device)
        model = UNet(22).to(device)

        model.load_state_dict(checkpoint['model'])
        return model, checkpoint['loss']


def replace_tensor_value_(tensor, a, b):
    tensor[tensor == a] = b
    return tensor


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-7)
    image.mul_(255)
    return image.type(torch.uint8).permute(1, 2, 0)


def array1d_to_pil_image(array):
    pil_out = Image.fromarray(array.astype(np.uint8), mode='P')
    pil_out.putpalette(PALETTE)
    return pil_out


def visualize(image, mask):
    image = normalize_image(image)
    mask = array1d_to_pil_image(mask.numpy())

    f, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].imshow(image)
    ax[1].imshow(mask)


def seg_classes(dataset):
    s = set()
    for img, mask in dataset:
        s.update(list(mask.unique().numpy()))
    return s


def plot_images(dataset, rows=2, cols=8, title=None):
    fig, axes = plt.subplots(rows, cols, dpi=150, figsize=(18, 12))
    fig.subplots_adjust(hspace=0)

    for r in range(rows):
        for c in range(0, cols, 2):
            image, mask = dataset[cols * r // 2 + c // 2]
            image = normalize_image(image)
            mask = array1d_to_pil_image(mask.numpy())
            axes[r, c].imshow(image)
            axes[r, c].axis('off')
            axes[r, c + 1].imshow(mask)
            axes[r, c + 1].axis('off')

