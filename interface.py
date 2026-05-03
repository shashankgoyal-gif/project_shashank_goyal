# interface.py

# Map your model class
from model import MushroomResNet18 as TheModel

# Map your training function
from train import train_mushroom_model as the_trainer

# Map your batch prediction function
from predict import classify_mushrooms as the_predictor

# Map your custom dataset class
from dataset import MushroomDataset as TheDataset

# Map your dataloader generator
from dataset import get_mushroom_dataloader as the_dataloader

# Map hyperparameters from config
from config import BATCH_SIZE as the_batch_size
from config import NUM_EPOCHS as total_epochs