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