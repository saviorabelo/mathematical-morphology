# -*- coding: utf-8 -*-
# Import library
import os
import cv2
import math
import numpy as np
from scipy import ndimage
from skimage.io import imread, imsave
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import disk
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
# Import functions
from functions import show_images_ws
from functions import threshold_mean


# Directory
dir_in = './Dataset/acer_negundo/'
dir_out = './Results KMeans/acer_negundo/'
my_list = [x for x in os.listdir(dir_in) if x.endswith('.jpg')]

for file_name in my_list:

    image = imread(dir_in+file_name)

    # KMeans
    (m ,n) = image.shape[:2]
    img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_vector = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters = 2)
    labels = kmeans.fit_predict(image_vector)
    quant = kmeans.cluster_centers_.astype('uint8')[labels]
    aux = quant.reshape((m, n, 3))
    img_kmeans = cv2.cvtColor(aux, cv2.COLOR_LAB2BGR)
    img_kmeans = rgb2gray(img_kmeans)

    otsu = threshold_otsu(img_kmeans)
    img_bin = ((img_kmeans > otsu) * 255).astype('uint8')

    img_close = morphology.closing(img_bin, disk(11))
    img_aux = img_bin - img_close
    img_ero = morphology.erosion(img_aux, disk(4))
    img_dil = morphology.dilation(img_ero, disk(10))

    _, n_comp = ndimage.label(img_dil)
    print('Number of components:', n_comp)

    # Plot
    fig = plt.figure(figsize=(10,10), dpi=80)
    a = fig.add_subplot(3,3,1); a.axis('off'); plt.imshow(image, cmap='gray'); a.set_title('Original')
    a = fig.add_subplot(3,3,2); plt.imshow(img_kmeans, cmap='gray'); a.set_title('KMeans'); a.axis('off')
    a = fig.add_subplot(3,3,3); plt.imshow(img_close, cmap='gray'); a.set_title('Closing'); a.axis('off')
    a = fig.add_subplot(3,3,4); plt.imshow(img_aux, cmap='gray'); a.set_title('Watershed - Closing'); a.axis('off')
    a = fig.add_subplot(3,3,5); plt.imshow(img_ero, cmap='gray'); a.set_title('Erosion'); a.axis('off')
    a = fig.add_subplot(3,3,6); plt.imshow(img_dil, cmap='gray'); a.set_title('Dilation'); a.axis('off')
    plt.tight_layout()
    plt.savefig(dir_out+file_name)
    #plt.show()