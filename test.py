import argparse
from PIL import Image
import matplotlib.pyplot as plt
from utils import Checkpoint, array1d_to_pil_image
from config import *
import torch
from torchvision import transforms


parser = argparse.ArgumentParser()

parser.add_argument('--image_path', type=str, required=True, help='input image path')
parser.add_argument('--checkpoint_path', type=str, required=True, help='checkpoint path')

args = parser.parse_args()

model, loss = Checkpoint.load_model(args.checkpoint_path)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

img = Image.open(args.image_path)

img = transform(img).to(device)
with torch.no_grad():
    outputs = model(img.unsqueeze(0).to('cuda'))

mask = outputs.squeeze(0).argmax(axis=0)
mask = array1d_to_pil_image(mask.cpu().numpy())
plt.imshow(mask)
