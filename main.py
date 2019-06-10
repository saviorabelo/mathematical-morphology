# coding: utf-8
# Import library
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.ndimage import label
from skimage import data
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import morphology
from skimage.morphology import square, rectangle, disk


file_name = '3303_lg.tiff'
folha = rgb2gray(imread(file_name, as_grey=True))

#folha = morphology.opening(folha, disk(5))
cont_ext = morphology.dilation(folha, disk(3)) - folha


# Exibe imagens
def show_images(image, markers, watershed):
    fig = plt.figure(figsize=(20,20))
    a = fig.add_subplot(1,3,1)
    plt.imshow(image, cmap=plt.cm.gray)
    a.set_title('Original')
    plt.axis('off')

    a = fig.add_subplot(1,3,2)
    plt.imshow(markers, cmap=plt.cm.gray)
    a.set_title('Marcadores')
    plt.axis('off')

    a = fig.add_subplot(1,3,3)
    plt.imshow(watershed,cmap=plt.cm.nipy_spectral, interpolation='nearest')
    a.set_title('Watershed')
    plt.axis('off')
    

fig = plt.figure(figsize=(10,10))
a = fig.add_subplot(1,1,1)
plt.imshow(folha, cmap=plt.cm.gray)
a.set_title('Original')
plt.axis('off')
