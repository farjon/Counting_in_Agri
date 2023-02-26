import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_image_metrics(image_path):
    # Load image
    image = io.imread(image_path)

    # Convert image to grayscale
    gray_image = rgb2gray(image)

    # Calculate SSIM
    ssim_score = ssim(image, gray_image, multichannel=True)

    # Calculate PSNR
    psnr_score = psnr(image, gray_image, data_range=image.max() - image.min())

    # Return metrics
    return {'SSIM': ssim_score, 'PSNR': psnr_score}



def calculate_color_distribution(image_path, num_bins=256):
    # Load image
    image = io.imread(image_path)

    # Calculate color histogram
    histogram, _ = np.histogramdd(image.reshape(-1, image.shape[-1]), bins=num_bins)
    histogram = histogram / np.sum(histogram)

    # Flatten histogram and convert to list
    histogram = histogram.flatten().tolist()

    # Return color distribution
    return histogram
