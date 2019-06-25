# -*- coding: utf-8 -*-
# Import library
import os
import math
import numpy as np
from scipy import ndimage
from skimage.io import imread, imsave
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.filters import threshold_otsu
# Import functions
from functions import show_images_ws
from functions import threshold_mean

# Directory
dir_in = './Dataset/acer_negundo/'
dir_out = './Results Watershed/acer_negundo/'
my_list = [x for x in os.listdir(dir_in) if x.endswith('.jpg')]

for file_name in my_list:

    img = imread(dir_in+file_name)
    image = rgb2gray(img)
    image_ext = morphology.dilation(image, disk(5)) - image

    # Watershed
    m,n = image.shape
    markers = np.zeros([m,n])
    m = math.floor(m/2) # Center
    n = math.floor(n/2)
    markers[20:40,20:40] = 200
    markers[m:m+20,n:n+20] = 100
    ws = morphology.watershed(image_ext, markers)

    # Plot
    #show_images_ws(img, 255-image_ext, markers, ws)

    #img_bin, mean = threshold_mean(ws)
    otsu = threshold_otsu(ws)
    img_bin = ((ws > otsu) * 255).astype('uint8')

    img_close = morphology.closing(img_bin, disk(11))
    img_aux = img_bin - img_close
    img_ero = morphology.erosion(img_aux, disk(4))
    img_dil = morphology.dilation(img_ero, disk(10))

    _, n_comp = ndimage.label(img_dil)
    print('Number of components:', n_comp)

    # Plot
    fig = plt.figure(figsize=(10,10), dpi=80)
    a = fig.add_subplot(3,3,1); a.axis('off'); plt.imshow(img, cmap='gray'); a.set_title('Original')
    a = fig.add_subplot(3,3,2); plt.imshow(ws, cmap='gray'); a.set_title('Watershed'); a.axis('off')
    a = fig.add_subplot(3,3,3); plt.imshow(img_close, cmap='gray'); a.set_title('Closing'); a.axis('off')
    a = fig.add_subplot(3,3,4); plt.imshow(img_aux, cmap='gray'); a.set_title('Watershed - Closing'); a.axis('off')
    a = fig.add_subplot(3,3,5); plt.imshow(img_ero, cmap='gray'); a.set_title('Erosion'); a.axis('off')
    a = fig.add_subplot(3,3,6); plt.imshow(img_dil, cmap='gray'); a.set_title('Dilation'); a.axis('off')
    plt.tight_layout()
    plt.savefig(dir_out+file_name)
    #plt.show()