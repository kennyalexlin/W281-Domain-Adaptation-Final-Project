import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp 
import cv2 as cv
from typing import Tuple
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor  
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from typing import Dict, List, Tuple

def standardize_image(image, target_size=(300, 300), method="pad", color=(0, 0, 0)):
    """
    Standardize an image by resizing it to the given target size using the specified method.
    
    Args:
        image (np.array): Input image.
        target_size (tuple): Target size (width, height) for resizing.
        method (str): Method for resizing - 'pad' for padding, 'crop' for center cropping.
        color (tuple): Color for padding (default is black for grayscale).

    Returns:
        np.array: Standardized image.
    """
    h, w = image.shape[:2]

    if method == "pad":
        # Add padding to make the image square
        if h > w:
            pad = (h - w) // 2
            padded_image = cv.copyMakeBorder(image, 0, 0, pad, h - w - pad, cv.BORDER_CONSTANT, value=color)
        else:
            pad = (w - h) // 2
            padded_image = cv.copyMakeBorder(image, pad, w - h - pad, 0, 0, cv.BORDER_CONSTANT, value=color)
        
        # Resize to the target size
        return cv.resize(padded_image, target_size, interpolation=cv.INTER_AREA)

    elif method == "crop":
        # Center crop the image to make it square
        min_dim = min(h, w)
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        cropped_image = image[start_h:start_h + min_dim, start_w:start_w + min_dim]
        
        # Resize to the target size
        return cv.resize(cropped_image, target_size, interpolation=cv.INTER_AREA)

    else:
        raise ValueError("Invalid method. Choose 'pad' or 'crop'.")


def load_split_images(
    img_dir,
    train_domains = ["amazon", "caltech10"], 
    test_val_split = 0.5,
    seed = 888) -> Tuple[Tuple, Tuple, Tuple]:
    # NOTE: no longer used. Only necessary prior to images being standardized.
    """ Loops through a directory and returns a tuple for train, val, and test splits respectively.
        Each tuple is of the form (domains, labels, image_paths, images)
        
        Args
            img_dir: path to image directory
            train_domains: domains to use in the train set
                remaining domains are used in the test and validation sets
            test_val_split: proportion of the test and val images to use in the test set
                the proportion to use in the val set is (1 - test_val_split)
            seed: random seed for shuffling test_val_split
        
        Returns
            tuple of Train, Val, and Test splits, each of the form (domains (np.array[str], labels (np.array[str]), image_paths (np.array[str]), images list) \
            with dtypes (np.array[str], np.array[str], np.array[str], list[np.array])
    """
    
    # TODO - standardize image size in train, test, and validation sets
    np.random.seed(seed)

    domains = []
    labels = []
    image_paths = []
    for domain in sorted(os.listdir(img_dir)):
        domain_path = os.path.join(img_dir, domain)
        if domain == '.DS_Store':
                continue
        for label in sorted(os.listdir(domain_path)):
            if label == '.DS_Store':
                continue
            label_path = os.path.join(domain_path, label)
            for filename in sorted(os.listdir(label_path)):
                if filename == '.DS_Store':
                    continue
                img_path = os.path.join(label_path, filename)
                
                domains.append(domain)
                labels.append(label)
                image_paths.append(img_path)
    
    domains = np.array(domains)
    labels = np.array(labels)
    image_paths = np.array(image_paths)
    
    train_bool_arr = np.isin(domains, train_domains)
    idx = np.arange(train_bool_arr.sum())
    np.random.shuffle(idx)
    
    train_domains = domains[train_bool_arr][idx]
    train_labels = labels[train_bool_arr][idx]
    train_image_paths = image_paths[train_bool_arr][idx]
    train_images = [plt.imread(path) for path in train_image_paths]
    
    test_val_domains = domains[~train_bool_arr]
    test_val_labels = labels[~train_bool_arr]
    test_val_image_paths = image_paths[~train_bool_arr]
    
    idx = np.arange(len(test_val_image_paths))
    np.random.shuffle(idx)
    test_idx = idx[:round(len(test_val_image_paths) * test_val_split)]
    val_idx = idx[round(len(test_val_image_paths) * test_val_split):]
    
    val_domains = test_val_domains[val_idx]
    val_labels = test_val_labels[val_idx]
    val_image_paths = test_val_image_paths[val_idx]
    val_images = [plt.imread(path) for path in val_image_paths]
    
    test_domains = test_val_domains[test_idx]
    test_labels = test_val_labels[test_idx]
    test_image_paths = test_val_image_paths[test_idx]
    test_images = [plt.imread(path) for path in test_image_paths]
    
    return (train_domains, train_labels, train_image_paths, train_images), (val_domains, val_labels, val_image_paths, val_images), (test_domains, test_labels, test_image_paths, test_images)


def load_images_by_domain(img_dir, target_size=(300, 300), method="pad", seed=888):
    """
    Load all images from a directory and organize them by domain, with standardization.

    Args:
        img_dir (str): Path to the image directory.
        target_size (tuple): Target size for image standardization.
        method (str): Standardization method ('pad' or 'crop').
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary where keys are domains, and values are (images, labels) tuples.
    """
    np.random.seed(seed)
    data_by_domain = {}

    for domain in sorted(os.listdir(img_dir)):
        if domain == '.DS_Store':
            continue
        domain_path = os.path.join(img_dir, domain)
        images = []
        labels = []

        for label in sorted(os.listdir(domain_path)):
            if label == '.DS_Store':
                continue
            label_path = os.path.join(domain_path, label)

            for filename in sorted(os.listdir(label_path)):
                if filename == '.DS_Store':
                    continue
                img_path = os.path.join(label_path, filename)
                image = cv.imread(img_path)
                if image is not None:
                    # Convert to grayscale if needed, then standardize
                    standardized_image = standardize_image(image, target_size, method)
                    images.append(standardized_image)
                    labels.append(label)

        data_by_domain[domain] = (np.array(images), np.array(labels))

    return data_by_domain

def split_images(
    data_by_domain: Dict[str, Tuple[np.ndarray, np.ndarray]],
    train_domains: List[str] = ["amazon", "caltech10"],
    test_domains: List[str] = ["dslr"],
    train_split: float = 0.6,
    val_split: float = 0.2,
    use_train_for_test: bool = False,
    test_split: float = 0.2,
    seed: int = 888,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split images into train, validation, and test sets.

    Args:
        data_by_domain (dict): Output of `load_images_by_domain`.
        train_domains (list): List of domains to include in the training set.
        test_domains (list): List of domains to include in the test set.
        train_split (float): Proportion of training data within the train domains.
        val_split (float): Proportion of validation data within the train domains.
        use_train_for_test (bool): Whether to use part of the train domains as the test set.
        test_split (float): Proportion of train domains to reserve for testing if `use_train_for_test` is True.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Train, validation, and test splits as dictionaries of (images, labels).
    """
    np.random.seed(seed)

    # Initialize containers
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    test_images, test_labels = [], []

    # Validate split proportions
    if use_train_for_test:
        total_split = train_split + val_split + test_split
        if not math.isclose(total_split, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"train_split + val_split + test_split must sum to 1.0 when use_train_for_test is True. "
                f"Current sum is {total_split}."
            )
    else:
        total_split = train_split + val_split
        # Allow test_split=1.0 when no training/validation is needed
        if test_split == 1.0 and train_split == 0.0 and val_split == 0.0:
            total_split = 1.0
        elif not math.isclose(total_split, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"train_split + val_split must sum to 1.0 when use_train_for_test is False, "
                f"or test_split must equal 1.0 when train_split and val_split are 0. "
                f"Current sum is {total_split}."
            )

    # Function to split data
    def split_data(images: np.ndarray, labels: np.ndarray, splits: Tuple[float, float, float] = (1.0, 0.0, 0.0)):
        idx = np.random.permutation(len(images))
        n_total = len(images)
        n_train = int(n_total * splits[0])
        n_val = int(n_total * splits[1])
        n_test = n_total - n_train - n_val  # Ensure all samples are used

        train_imgs = images[idx[:n_train]]
        train_lbls = labels[idx[:n_train]]

        val_imgs = images[idx[n_train:n_train + n_val]]
        val_lbls = labels[idx[n_train:n_train + n_val]]

        test_imgs = images[idx[n_train + n_val:]]
        test_lbls = labels[idx[n_train + n_val:]]

        return train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls

    # Handle train domains
    for domain in train_domains:
        images, labels = data_by_domain[domain]
        if use_train_for_test:
            splits = (train_split, val_split, test_split)
        else:
            splits = (train_split, val_split, 0.0)
        t_imgs, t_lbls, v_imgs, v_lbls, te_imgs, te_lbls = split_data(images, labels, splits)
        train_images.append(t_imgs)
        train_labels.append(t_lbls)
        val_images.append(v_imgs)
        val_labels.append(v_lbls)
        if use_train_for_test:
            test_images.append(te_imgs)
            test_labels.append(te_lbls)

    # Handle test domains
    if not use_train_for_test:
        for domain in test_domains:
            images, labels = data_by_domain[domain]
            test_images.append(images)
            test_labels.append(labels)

    # Concatenate all splits
    def concatenate_splits(splits_list: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(splits_list) if splits_list else np.array([], dtype=np.float32)

    train_images = concatenate_splits(train_images)
    train_labels = concatenate_splits(train_labels)
    val_images = concatenate_splits(val_images)
    val_labels = concatenate_splits(val_labels)
    test_images = concatenate_splits(test_images)
    test_labels = concatenate_splits(test_labels)

    # **Removed label conversion to integers**

    return (
        {"images": train_images, "labels": train_labels},
        {"images": val_images, "labels": val_labels},
        {"images": test_images, "labels": test_labels},
    )

def create_image_fig(images, images_per_row=3, titles=None):
    """
    returns a fig containing an arbitrary number of images. 
    
    
    Args
        images: list of images to display
    Returns
        figure containing the images
    """
    # get number of rows needed
    num_images = len(images)
    num_rows = math.ceil(num_images / images_per_row)

    if titles is None:
        titles = [f"Img {i+1}" for i in range(num_images)]
    # create subplot grid
    fig = sp.make_subplots(
        rows=num_rows, 
        cols=images_per_row, 
        subplot_titles=titles,
        horizontal_spacing=0,
        vertical_spacing=0.1
    )

    # add images to each subplot
    for idx, img in enumerate(images):
        row = (idx // images_per_row) + 1
        col = (idx % images_per_row) + 1
        fig.add_trace(go.Image(z=img), row=row, col=col)

    fig.update_layout(
        showlegend=False,
        height=200 * num_rows,
        width=250 * images_per_row,
        margin=dict(l=5, r=5, t=30, b=5)
    )

    # remove axis ticks
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    return fig

def visualize_corners(img, maxCorners=25, qualityLevel=0.2, minDistance=5, apply_canny=False, show_canny=False):
    """ Applies shi-tomasi corner detection
        
        Args
            img: grayscale version of image
            maxCorners: maximum amount of corners to find. corners are sorted by quality
                and only the top n corners are retrieved
            qualityLevel: how "corner-like" a point must be before it's considered a corner.
                in shi-tomasi corner detection, a corner is edge-like if there's a high intensity
                derivative in the x AND y directions.
            minDistance: the minimum distance between each corner that's identified. If two corners
                are identified but are too close, the one with a higher qualityLevel is kept
        Returns
            list of image coordinates where corners are
            also displays a plot of corners on the image
            
    """
    
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if apply_canny:
        gray_img = cv.Canny(gray_img, 100, 200)
    corners = cv.goodFeaturesToTrack(gray_img, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance, useHarrisDetector=False)
    if show_canny:
        img = gray_img
    fig = px.imshow(img, width=300, height=300)
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_layout(margin=dict(l=5, r=5, t=30, b=5))
    
    if corners is not None:
        corners = corners[:,0,:]
        scatter = go.Scatter(
            x=[x for x, y in corners], 
            y=[y for x, y in corners],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Points'
        )
        fig.add_trace(scatter)
    return corners, fig

def get_mean_intensity_rgb(imgs: np.ndarray) -> list:
    """Returns the mean intensity of each color channel (Red, Green, Blue) for a list of images.

    Args:
        imgs (list or np.ndarray): A list of images or a single image represented as numpy arrays. 
                                   Each image should have shape (height, width, channels).
    Returns:
        list: A list containing the mean intensity of each color channel (Red, Green, Blue), 
              rounded to 2 decimal places.
    """
    # If the input is a single image, convert it to a list of images
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]

    # Loop over each color channel (Red, Green, Blue)
    mean_intensities_all_imgs = []
    for i in range(3):
        mean_intensities_this_color = []
        for img in imgs:
            if np.ndim(img) == 3: # We need to skip grayscale images for this (one of the laptop images was throwing an error bc of this)
                mean_intensity = np.mean(img[:, :, i])
                mean_intensities_this_color.append(mean_intensity)
            else:
                continue
        mean_intensities_all_imgs.append(np.mean(mean_intensities_this_color))

    return [round(i, 2) for i in mean_intensities_all_imgs]


def get_mean_intensity_grayscale(imgs: np.ndarray) -> list:
    """Returns the mean pixel intensity of images. 
    
    Args:
        imgs (list or np.ndarray): A list of images or a single image represented as numpy arrays. 
                                   Each image should have shape (height, width, channels) or (height, width).
    Returns:
        list: A list containing the mean intensity of each image, rounded to 2 decimal places.
    """
    # If the input is a single image, convert it to a list of images
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]

    mean_img_intensities = []
    for img in imgs:
        if np.ndim(img) == 3:  # Convert RGB to grayscale
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mean_img_intensities.append(np.mean(img))
    
    return round(np.mean(mean_img_intensities), 2)



def plot_color_histogram(imgs, label, intensity_filter: list = [0, 256]) -> plt.figure:
    """Plots the color histogram of an image and returns the plot figure.

    This function generates and plots the histogram for each color channel (red, green, and blue) of the input image.
    It also fills the area under the histogram curve with a semi-transparent color. The x-axis represents the intensity
    values (0-255), and the y-axis represents the frequency of these intensity values.

    Args:
        img (numpy.ndarray): The input image for which the color histogram is to be plotted.
        label (str): The label or title for the histogram plot.
        pixel_value_range (list): The range of pixel values to consider when plotting the histogram. 
                                  Default is [0, 256].
            This is a cheap easy way to omit the all white pixels, might want to improve this later on. 

    Returns:
        (matplotlib.figure.Figure): The figure object containing the histogram plot.
    """
    # If the input is a single image, convert it to a list of images
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]

    colors = ('r', 'g', 'b')

    plt.figure(figsize=(5, 5))
    for i, color_channel in enumerate(colors):
        histr = cv.calcHist(
            images = imgs,
            channels = [i],
            mask = None, 
            histSize = [intensity_filter[1] - intensity_filter[0]], 
            ranges = intensity_filter,
        )
        plt.plot(histr, color = color_channel)
        plt.fill_between(
            range(len(histr)),
            histr[:, 0], 
            color=color_channel, 
            alpha=0.1
        )

    plt.xlim(intensity_filter)
    plt.title(f'Color Histogram - {label}')
    return plt.gca()

def convert_to_grayscale(image):
    """
    Convert an image to grayscale if it's in RGB/BGR/RGBA format.

    Args:
        image (np.array): Input image.

    Returns:
        np.array: Grayscale image.
    """
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            # Convert RGBA to Grayscale
            image = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
        else:
            # Convert BGR to Grayscale
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif len(image.shape) == 4:
        # Handle images with multiple channels, e.g., batch size
        # This depends on your data structure; adjust accordingly
        raise ValueError(f"Unexpected image shape: {image.shape}")
    return image

def compute_multiscale_lbp_features(image, PR_combinations):
    """
    Compute multi-scale LBP features from an image.
    Args:
        image (np.array): Grayscale image
        PR_combinations (list of tuples): List of (P, R) combinations to use

    Returns:
        multiscale_features (np.array): Concatenated LBP histograms for all (P, R) combinations
    """
    multiscale_features = []
    for P, R in PR_combinations:
        lbp = local_binary_pattern(image, P, R, method="ror")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        multiscale_features.extend(lbp_hist)
    return np.array(multiscale_features)

def extract_lbp_features(split_data, PR_combinations):
    """
    Extract LBP features for all images in a data split.

    Args:
        split_data (dict): Dictionary with keys 'images' and 'labels'.
        PR_combinations (list of tuples): LBP (P, R) combinations.

    Returns:
        pd.DataFrame: DataFrame containing LBP features and labels.
    """
    features = []
    labels = split_data['labels']

    print(f"Extracting LBP features from {len(split_data['images'])} images...")

    for img in tqdm(split_data['images']):
        # Ensure the image is grayscale and in uint8 format
        img_gray = convert_to_grayscale(img)
        img_gray = (img_gray * 255).astype(np.uint8) if img_gray.max() <= 1.0 else img_gray.astype(np.uint8)

        # Extract LBP features
        lbp_features = compute_multiscale_lbp_features(img_gray, PR_combinations)
        features.append(lbp_features)

    # Create DataFrame
    feature_array = np.array(features)
    feature_columns = [f"LBP_feature_{i}" for i in range(feature_array.shape[1])]
    df = pd.DataFrame(feature_array, columns=feature_columns)
    df['label'] = labels
    return df

def compute_glcm_features(image, distances, angles):
    """
    Compute GLCM features for an image at multiple distances and angles.
    Args:
        image (np.array): Grayscale image
        distances (list): List of distances for GLCM calculation
        angles (list): List of angles (in radians) for GLCM calculation

    Returns:
        features (np.array): Concatenated GLCM features for all distances and angles
    """
    glcm_features = []
    for distance in distances:
        for angle in angles:
            glcm = graycomatrix(image, distances=[distance], angles=[angle], levels=256, symmetric=True, normed=True)
            glcm_features.extend([
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'dissimilarity')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'correlation')[0, 0],
            ])
    return np.array(glcm_features)

def extract_glcm_features_split(split_data, glcm_distances, glcm_angles):
    """
    Extract GLCM features for all images in a data split.

    Args:
        split_data (dict): Dictionary with keys 'images' and 'labels'.
        glcm_distances (list): GLCM distances.
        glcm_angles (list): GLCM angles.

    Returns:
        pd.DataFrame: DataFrame containing GLCM features and labels.
    """
    features = []
    labels = split_data['labels']

    print(f"Extracting GLCM features from {len(split_data['images'])} images...")

    for img in tqdm(split_data['images']):
        # Ensure the image is grayscale and in uint8 format
        img_gray = convert_to_grayscale(img)
        img_gray = (img_gray * 255).astype(np.uint8) if img_gray.max() <= 1.0 else img_gray.astype(np.uint8)

        # Extract GLCM features
        glcm_features = compute_glcm_features(img_gray, glcm_distances, glcm_angles)
        features.append(glcm_features)

    # Create DataFrame
    feature_array = np.array(features)
    feature_columns = [f"GLCM_feature_{i}" for i in range(feature_array.shape[1])]
    df = pd.DataFrame(feature_array, columns=feature_columns)
    df['label'] = labels
    return df

def compute_gabor_features(image, frequencies, angles):
    """
    Compute Gabor features for an image at multiple frequencies and angles.
    
    Args:
        image (np.array): Grayscale image
        frequencies (list): List of frequencies for Gabor filter
        angles (list): List of angles (in radians) for Gabor filter

    Returns:
        features (np.array): Concatenated Gabor features for all frequencies and angles
    """
    gabor_features = []
    for frequency in frequencies:
        for angle in angles:
            real, imag = gabor(image, frequency=frequency, theta=angle)
            magnitude = np.sqrt(real**2 + imag**2)
            
            # Compute statistical features for the magnitude response
            gabor_features.extend([
                np.mean(magnitude),  # Mean
                np.std(magnitude),   # Standard deviation
                np.median(magnitude),  # Median
                np.sum(magnitude)    # Sum (Energy)
            ])
    return np.array(gabor_features)

def extract_gabor_features_split(split_data, gabor_frequencies, gabor_angles):
    """
    Extract Gabor features for all images in a data split.

    Args:
        split_data (dict): Dictionary with keys 'images' and 'labels'.
        gabor_frequencies (list): Gabor frequencies.
        gabor_angles (list): Gabor angles.

    Returns:
        pd.DataFrame: DataFrame containing Gabor features and labels.
    """
    features = []
    labels = split_data['labels']

    print(f"Extracting Gabor features from {len(split_data['images'])} images...")

    for img in tqdm(split_data['images']):
        # Convert to grayscale
        img_gray = convert_to_grayscale(img)

        # Ensure image is in uint8 format
        if img_gray.dtype != np.uint8:
            if img_gray.max() <= 1.0:
                img_gray = (img_gray * 255).astype(np.uint8)
            else:
                img_gray = img_gray.astype(np.uint8)

        # Extract Gabor features
        gabor_feat = compute_gabor_features(img_gray, gabor_frequencies, gabor_angles)
        features.append(gabor_feat)

    # Create DataFrame
    feature_array = np.array(features)
    feature_columns = [f"Gabor_feature_{i}" for i in range(feature_array.shape[1])]
    df = pd.DataFrame(feature_array, columns=feature_columns)
    df['label'] = labels
    return df

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Custom Dataset for loading images and labels.

        Args:
            images (list or np.array): List or array of images.
            labels (list or np.array): Corresponding integer labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Ensure label is an integer
        if not isinstance(label, int):
            raise TypeError(f"Label '{label}' at index {idx} is not an integer.")

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def extract_resnet_features_split(split_data, split_name, batch_size=16, device=None, num_classes=10, int_to_label=None):
    """
    Extract ResNet101 penultimate layer features for all images in a data split.

    Args:
        split_data (dict): Dictionary with keys 'images' and 'labels'.
        split_name (str): Name of the data split ('train', 'val', 'test').
        batch_size (int): Batch size for DataLoader.
        device (torch.device): PyTorch device (CPU or GPU). If None, defaults to CUDA if available.
        num_classes (int): Number of classes for the ResNet model.
        int_to_label (dict): Mapping from integer labels to class names.

    Returns:
        pd.DataFrame: DataFrame containing ResNet features and labels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations for ResNet101
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create DataLoader using CustomDataset
    dataset = CustomDataset(split_data["images"], split_data["labels"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load ResNet101 pre-trained model
    model = models.resnet101(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Identity()  # Remove the final classification layer
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    features = []
    labels = split_data['labels']

    print(f"Extracting ResNet features from {len(split_data['images'])} images for '{split_name}' split...")

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=f"Extracting ResNet Features ({split_name} split)"):
            images = images.to(device)
            batch_features = model(images)
            features.append(batch_features.cpu().numpy())

    # Combine all batches into a single array
    feature_array = np.concatenate(features, axis=0)

    # Create descriptive feature names
    feature_columns = [f"ResNet_feature_{i}" for i in range(feature_array.shape[1])]

    # Create DataFrame
    df = pd.DataFrame(feature_array, columns=feature_columns)
    if int_to_label is not None:
        df['label'] = [int_to_label[label] for label in labels]
    else:
        df['label'] = labels

    print(f"ResNet feature extraction completed for '{split_name}' split.")

    return df


def compute_resnet_features(dataset_splits, splits_to_process, batch_size=16, save_csv=True, output_dir="features"):
    """
    Generalized function to process and extract ResNet features for given dataset splits and save them to a 'features' subdirectory.

    Args:
        dataset_splits (dict): Dictionary containing dataset splits ('train', 'val', 'test').
        splits_to_process (list): List of dataset splits to process (e.g., ['train', 'val', 'test']).
        batch_size (int): Batch size for feature extraction.
        save_csv (bool): Whether to save the resulting DataFrame to a CSV file.
        output_dir (str): Directory where CSV files will be saved (default is 'features').

    Returns:
        dict: Dictionary with DataFrames of extracted features for each processed split.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect all unique labels from the splits
    all_labels = []
    for split in dataset_splits.values():
        all_labels.extend(split['labels'])

    unique_classes = sorted(set(all_labels))
    label_to_int = {class_name: idx for idx, class_name in enumerate(unique_classes)}
    int_to_label = {idx: class_name for class_name, idx in label_to_int.items()}
    print("Label to integer mapping:", label_to_int)

    # Function to convert labels to integers
    def convert_labels_to_int(split_data):
        return {
            'images': split_data['images'],
            'labels': [label_to_int[label] for label in split_data['labels']]
        }

    # Extract features for each requested split
    extracted_features = {}
    for split_name in splits_to_process:
        if split_name not in dataset_splits:
            print(f"Skipping '{split_name}' as it is not in the dataset splits.")
            continue

        print(f"Processing {split_name} split...")

        # Convert labels to integers for this split
        split_data_int = convert_labels_to_int(dataset_splits[split_name])

        # Extract ResNet features for the split
        df = extract_resnet_features_split(
            split_data_int,
            split_name,
            batch_size=batch_size,
            num_classes=len(unique_classes),
            int_to_label=int_to_label
        )

        # Save to 'features' subdirectory if required
        if save_csv:
            # Add domain information to the filename if available
            domain_prefix = dataset_splits[split_name].get("domain", "").lower()
            filename_prefix = f"{domain_prefix}_" if domain_prefix else ""
            output_path = os.path.join(output_dir, f"{filename_prefix}{split_name}_resnet_features.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {split_name} features to '{output_path}'.")

        extracted_features[split_name] = df

    return extracted_features

def get_orb_features(
    imgs,
    nfeatures=500,
    patchSize=10,
    scaleFactor=1.2,
    scoreType=0,
    n_clusters=200,
    kmeans=None,
):
    desc = []
    desc_flat = []
    orb = cv.ORB_create(
        nfeatures=nfeatures, 
        edgeThreshold=patchSize, 
        patchSize=patchSize,
        scaleFactor=scaleFactor,
        scoreType=scoreType
    )
    print("Getting ORB keypoints...")
    for idx, img in enumerate(tqdm(imgs)):
        # convert to grayscale only to reduce impact of differences in hue
        # try/except because some imgs are already in grayscale
        try:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
        except:
            pass
        keypoints = orb.detect(img)
        keypoints, descriptors = orb.compute(img, keypoints)
        to_add = []
        if descriptors is not None:
            to_add = descriptors.tolist()
        desc.append(to_add)
        desc_flat = desc_flat + to_add
    
    # if no kmeans is provided, fit a new one
    if kmeans is None:
        print(f"No kmeans was provided, so fitting a new one...")
        kmeans = KMeans(random_state=88, n_clusters=n_clusters)
        kmeans.fit(desc_flat)
        
    dense = [kmeans.predict(i) if len(i) else [] for i in desc]
    
    sparse = []
    for vw_dense in dense:
        vw_sparse = np.zeros(n_clusters)
        for vw in vw_dense:
            vw_sparse[vw] += 1
        sparse.append(vw_sparse)
        
    features = np.stack(sparse)
    
    transformer = TfidfTransformer()
    features = transformer.fit_transform(features).toarray()
    
    return features, kmeans