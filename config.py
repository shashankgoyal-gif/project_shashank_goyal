# Hyperparameters
DATA_DIR = 'data' 
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Image Resize Info
RESIZE_X = 256 #Resizing to 256>224 to create a buffer such that the code randomly crops 224 pixels out of 256
RESIZE_Y = 256
CROP_SIZE = 224
INPUT_CHANNELS = 3

# Normalization Stats of ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
