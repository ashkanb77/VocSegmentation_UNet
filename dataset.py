import torch
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from PIL import Image
import numpy as np
from utils import replace_tensor_value_
from config import IMAGENET_MEAN, IMAGENET_STD


class VOCSeg(VOCSegmentation):
    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    @property
    def masks(self):
        return self.targets

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        img = np.array(img)
        mask = transforms.PILToTensor()(Image.open(self.masks[index]))
        mask = replace_tensor_value_(mask.squeeze(0).long(), 255, 21).numpy()

        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']

        img = torch.from_numpy(img) / 255
        img = img.permute(2, 0, 1)
        img = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(img)
        return img, torch.from_numpy(mask)
