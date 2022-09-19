import torch

LEARNING_RATE = 0.001
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 70
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DATASET_DIR = 'dataset'
CHECK_DIR = 'checkpoints'
MODEL_NAME = 'UNet'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
