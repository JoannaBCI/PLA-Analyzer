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

def segment_nuclei(image):
    blurred = filters.gaussian(image, sigma=2)
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    labeled = measure.label(binary)
    return labeled

def count_puncta(pla_image, labeled_nuclei):
    blurred = filters.gaussian(pla_image, sigma=1)
    thresh = filters.threshold_otsu(blurred)
    puncta_mask = blurred > thresh
    puncta_labeled = measure.label(puncta_mask)
    counts = {}
    for nucleus_id in range(1, labeled_nuclei.max() + 1):
        nucleus_mask = labeled_nuclei == nucleus_id
        dots_in_nucleus = puncta_labeled[nucleus_mask]
        counts[nucleus_id] = len(np.unique(dots_in_nucleus)) - 1
    return counts

def make_plot(counts, cell_type)
    pass