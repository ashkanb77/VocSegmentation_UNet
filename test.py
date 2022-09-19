import numpy as np
import argparse
from PIL import Image
from utils import Checkpoint
from config import *
import torch
import albumentations as A

parser = argparse.ArgumentParser()

parser.add_argument('--image_path', type=str, required=True, help='input image path')
parser.add_argument('--question', type=str, required=True, help='input question')

args = parser.parse_args()

model, loss = Checkpoint.load_model(args.checkpoint_path)

img = Image.open(args.image_path)

transforms = A.Compose([
        A.Resize(512, 512)
])

with torch.no_grad():
    img = transforms(np.array(img)).to(device)
    outputs = model(torch.from_numpy(img).unsqueeze(0))

