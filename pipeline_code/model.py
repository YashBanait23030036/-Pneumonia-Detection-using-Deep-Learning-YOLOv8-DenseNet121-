import torch.nn as nn
from torchvision.models import densenet121

# Add this function to your model.py
def get_densenet121(num_classes=2):
    model = densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model
