import argparse
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
 
from skimage import filters, measure, segmentation, morphology, color
from skimage.io import imread, imsave
from scipy import ndimage as ndi

def load_image(path: str) -> np.ndarray:
    img = imread(path)
    if img.ndim == 3 and img.shape[2] not in (3, 4):
        img = img.max(axis=0)
    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = color.rgb2gray(img[..., :3])
        img = (img * 255).astype(np.float32)
    else:
        img = img.astype(np.float32)
    return img

def segment_nuclei(image: np.ndarray,
                   gaussian_sigma: float = 2.0,
                   min_nucleus_area_um2: float = 30.0,
                   pixel_size_um: float = 1.0) -> np.ndarray:
    blurred = filters.gaussian(image, sigma=gaussian_sigma)
      thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    binary = ndi.binary_fill_holes(binary)
    min_area_px = int(min_nucleus_area_um2 / (pixel_size_um ** 2))
    binary = morphology.remove_small_objects(binary, min_size=min_area_px)
    distance = ndi.distance_transform_edt(binary)
    coords = measure.peak_local_max(distance, footprint=np.ones((9, 9)), labels=binary)
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_peaks)
    labeled = segmentation.watershed(-distance, markers, mask=binary)
    return labeled