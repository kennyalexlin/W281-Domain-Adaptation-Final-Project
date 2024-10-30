import os, math
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp 
import cv2 as cv
from typing import Tuple

def standardize_image():
    # TODO - implement
    pass

def load_split_images(
    img_dir,
    train_domains = ["amazon", "caltech10"], 
    test_val_split = 0.5,
    seed = 888) -> Tuple[Tuple, Tuple, Tuple]:
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
        for label in sorted(os.listdir(domain_path)):
            label_path = os.path.join(domain_path, label)
            for filename in sorted(os.listdir(label_path)):
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
        height=300 * num_rows,
        width=900,
        margin=dict(l=5, r=5, t=30, b=5)
    )

    # remove axis ticks
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    return fig

def visualize_corners(img, maxCorners=25, qualityLevel=0.2, minDistance=5):
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
    corners = cv.goodFeaturesToTrack(gray_img, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance, useHarrisDetector=False)
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
    fig.show()
    return corners