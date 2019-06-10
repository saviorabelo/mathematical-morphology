# coding: utf-8
# Import library
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage import label
from skimage import data
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import morphology
from skimage.morphology import disk, square, rectangle
# Import functions
from functions import *

file_name = './Dataset Image/Alprazolam/4047_lg.jpg'
image = rgb2gray(imread(file_name))
image_ext = morphology.dilation(image, disk(3)) - image

# Watershed
markers = np.zeros(image.shape)
markers[90:110,20:40] = 200
markers[90:110,90:110] = 100
ws = morphology.watershed(image_ext, markers)

# Plot
show_images(255-image_ext, markers, ws)
