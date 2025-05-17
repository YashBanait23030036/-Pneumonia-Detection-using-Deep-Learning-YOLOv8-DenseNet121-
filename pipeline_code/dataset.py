import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with image filenames and labels.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])  # filename is in the first column
        image = Image.open(img_name).convert("RGB")  # Open image in RGB mode
        label = int(self.data_frame.iloc[idx, 1])  # Get the label (Normal or Pneumonia)

        if self.transform:
            image = self.transform(image)

        return image, label
