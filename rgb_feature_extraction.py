# rgb_feature_extraction.py

#### Load in data ####

import os
import cv2
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from utils import load_images_by_domain, split_images

# Define paths
img_dir = "../OfficeCaltechDomainAdaptation/images"

# Load images by domain
data_by_domain = load_images_by_domain(
    img_dir=img_dir,
    target_size=(300, 300),  # Standardized size
    method="pad",           # Use padding to maintain aspect ratio
    seed=888                # Seed for reproducibility
)

# Split images: Combine amazon and caltech10 into train/val/test
train_data, val_data, test_data = split_images(
    data_by_domain=data_by_domain,
    train_domains=["amazon", "caltech10"],  # Combine these for training and validation
    test_domains=[],                        # Use part of amazon and caltech10 for testing
    train_split=0.6,                        # 60% for training
    val_split=0.2,                          # 20% for validation
    use_train_for_test=True,                # Use part of train_domains for testing
    test_split=0.2,                         # 20% for testing
    seed=888                                # Seed for reproducibility
)

# Summary of splits
print(f"Train images: {len(train_data['images'])}, Train labels: {len(train_data['labels'])}")
print(f"Validation images: {len(val_data['images'])}, Validation labels: {len(val_data['labels'])}")
print(f"Test images: {len(test_data['images'])}, Test labels: {len(test_data['labels'])}")



from utils import extract_RGB_features

# Extract RGB features for each split
train_rgb_df = extract_RGB_features(train_data)
val_rgb_df = extract_RGB_features(val_data)
test_rgb_df = extract_RGB_features(test_data)

# Save RGB features to CSV
os.makedirs("features", exist_ok=True)
train_rgb_df.to_csv(os.path.join("features", "train_rgb_features.csv"), index=False)
val_rgb_df.to_csv(os.path.join("features", "val_rgb_features.csv"), index=False)
test_rgb_df.to_csv(os.path.join("features", "test_rgb_features.csv"), index=False)

print("RGB feature extraction and saving completed successfully!")