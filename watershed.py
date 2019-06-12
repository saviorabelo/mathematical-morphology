# coding: utf-8
# Import library
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import morphology
from skimage.morphology import disk, square, rectangle
# Import functions
from functions import *

#file_name = './Dataset Image/Alprazolam/3527_lg.jpg'
file_name = './Dataset Image/Domino/4424_lg.jpg'
img = imread(file_name)
image = rgb2gray(img)
image_ext = morphology.dilation(image, disk(3)) - image

# Watershed
markers = np.zeros(image.shape)
markers[10:30,20:40] = 200
markers[90:110,90:110] = 100 # Center
ws = morphology.watershed(image_ext, markers)

# Plot
show_images(img, 255-image_ext, markers, ws)
