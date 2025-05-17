import torch
import os

class Config:
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Image settings
        self.image_size = 224
        self.num_classes = 2

        # Training settings
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_workers = 2  # You can adjust based on your CPU

        # Paths
        self.csv_file = "D:/rsna_dataset/processed_images/final_balanced_dataset_cleaned.csv"
        self.train_csv = "D:/rsna_dataset/processed_images/train_split.csv"
        self.val_csv = "D:/rsna_dataset/processed_images/val_split.csv"
        self.image_root = "D:/rsna_dataset/processed_images/train_segmented"

        self.pneumonia_augmented_dir = "D:/rsna_dataset/cropped_pneumonia_regions_augmented"
        self.normal_dir = "D:/rsna_dataset/processed_images/train_segmented"

        # Checkpoints
        self.checkpoint_dir = "D:/rsna_dataset/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Weighted loss if needed
        self.class_weights = torch.tensor([1.0, 1.0])

        # Seed
        self.seed = 42
