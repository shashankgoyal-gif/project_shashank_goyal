# predict.py
import torch
from PIL import Image
from torchvision import transforms
from config import RESIZE_X, CROP_SIZE, MEAN, STD

def inferloader(list_of_img_paths):
    """Converts a list of raw image paths into a batch tensor."""
    transform = transforms.Compose([
        transforms.Resize(RESIZE_X),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    tensors = []
    for path in list_of_img_paths:
        img = Image.open(path).convert('RGB')
        tensors.append(transform(img))
        
    return torch.stack(tensors)

def classify_mushrooms(list_of_img_paths, model, class_names=['edible', 'poisonous']):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Convert paths to input suitable for the model
    mushroom_batch = inferloader(list_of_img_paths)
    mushroom_batch = mushroom_batch.to(device)
    
    # Predict the outcome
    with torch.no_grad():
        logits = model(mushroom_batch)
        _, preds = torch.max(logits, 1)
        
    # Map predictions to class names
    labels = [class_names[p] for p in preds]
    return labels