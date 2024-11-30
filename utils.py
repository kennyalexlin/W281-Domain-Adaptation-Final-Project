import os, math
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp 
import cv2 as cv
from typing import Tuple

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
        domain_path = os.path.join(img_dir, domain)
        images = []
        labels = []

        for label in sorted(os.listdir(domain_path)):
            label_path = os.path.join(domain_path, label)

            for filename in sorted(os.listdir(label_path)):
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
    data_by_domain,
    train_domains=["amazon", "caltech10"],
    test_domains=["dslr"],
    train_split=0.8,
    val_split=0.2,
    use_train_for_test=False,  # New argument to use part of train domains as test set
    test_split=0.0,            # Proportion of train domains to reserve for testing
    seed=888,
    combine_train=False,
):
    """
    Split images into train, validation, and test sets, ensuring train + val + test = 1.0.

    Args:
        data_by_domain (dict): Output of `load_images_by_domain`.
        train_domains (list): List of domains to include in the training set.
        test_domains (list): List of domains to include in the test set.
        train_split (float): Proportion of training data within the train domains.
        val_split (float): Proportion of validation data within the train domains.
        use_train_for_test (bool): Whether to use part of the train domains as the test set.
        test_split (float): Proportion of train domains to reserve for testing if `use_train_for_test` is True.
        seed (int): Random seed for reproducibility.
        combine_train (bool): Whether to combine all train domains into a single training set.

    Returns:
        tuple: Train, validation, and test splits as dictionaries of (images, labels).
    """
    # Ensure train + val + test = 1.0
    if use_train_for_test:
        total_split = train_split + val_split + test_split
        if not math.isclose(total_split, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Invalid splits: train_split ({train_split}), val_split ({val_split}), "
                f"and test_split ({test_split}) must sum to 1.0. Currently, they sum to {total_split}."
            )
    else:
        total_split = train_split + val_split
        if not math.isclose(total_split, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Invalid splits: train_split ({train_split}) and val_split ({val_split}) "
                f"must sum to 1.0 when not using train for test. Currently, they sum to {total_split}."
            )

    np.random.seed(seed)

    train_images, train_labels = [], []
    val_images, val_labels = []
    test_images, test_labels = [], []

    # Handle training domains
    for domain in train_domains:
        images, labels = data_by_domain[domain]
        idx = np.arange(len(images))
        np.random.shuffle(idx)

        if use_train_for_test:
            # Split train domain into train, validation, and test subsets
            test_split_point = int(len(images) * test_split)
            train_split_point = int(len(images[test_split_point:]) * train_split)

            test_images.extend(images[idx[:test_split_point]])
            test_labels.extend(labels[idx[:test_split_point]])

            train_images.append(images[idx[test_split_point:test_split_point + train_split_point]])
            train_labels.append(labels[idx[test_split_point:test_split_point + train_split_point]])

            val_images.append(images[idx[test_split_point + train_split_point:]])
            val_labels.append(labels[idx[test_split_point + train_split_point:]])
        else:
            # Standard train/val split without reserving for test
            split_point = int(len(images) * train_split)
            train_images.append(images[idx[:split_point]])
            train_labels.append(labels[idx[:split_point]])

            val_images.append(images[idx[split_point:]])
            val_labels.append(labels[idx[split_point:]])

    if combine_train:
        train_images = [img for domain_imgs in train_images for img in domain_imgs]
        train_labels = [lbl for domain_lbls in train_labels for lbl in domain_lbls]
        val_images = [img for domain_imgs in val_images for img in domain_imgs]
        val_labels = [lbl for domain_lbls in val_labels for lbl in domain_lbls]

    train_data = {"images": np.array(train_images), "labels": np.array(train_labels)}
    val_data = {"images": np.array(val_images), "labels": np.array(val_labels)}

    # Handle test domains (if not using train for test)
    if not use_train_for_test:
        for domain in test_domains:
            images, labels = data_by_domain[domain]
            test_images.extend(images)
            test_labels.extend(labels)

    test_data = {"images": np.array(test_images), "labels": np.array(test_labels)}

    return train_data, val_data, test_data

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

def compute_multiscale_lbp_features(image, PR_combinations):
    """
    compute multi-scale LBP features from an image.
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

def extract_resnet_penultimate_features(split_data, batch_size=16, device=None, num_classes=10):
    """
    Extract penultimate layer features from a ResNet101 model for a given dataset split.

    Args:
        split_data (dict): Dataset split containing "images" and "labels".
        batch_size (int): Batch size for DataLoader.
        device (torch.device): PyTorch device (CPU or GPU). If None, defaults to CUDA if available.
        num_classes (int): Number of classes for the ResNet model.

    Returns:
        np.array: Penultimate layer features.
        np.array: Corresponding labels.
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

    # Create DataLoader
    dataset = CustomDataset(split_data["images"], split_data["labels"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load ResNet101 pre-trained model
    model = models.resnet101(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)  # Replace final layer
    model = model.to(device)

    # Extract the penultimate layer
    penultimate_layer_model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
    penultimate_layer_model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in tqdm(dataloader, desc="Extracting Penultimate Features"):
            images = images.to(device)
            batch_features = penultimate_layer_model(images)
            features.append(batch_features.view(batch_features.size(0), -1).cpu().numpy())  # Flatten features
            labels.append(batch_labels.numpy())

    # Combine all batches into single arrays
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels
