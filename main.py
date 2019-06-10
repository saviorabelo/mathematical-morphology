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


file_name = './Dataset Image/Alprazolam/4047_lg.jpg'
image = rgb2gray(imread(file_name))

#image = morphology.opening(image, disk(5))
image_ext = morphology.dilation(image, disk(3)) - image

# Watershed
markers = np.zeros(image.shape)
markers[90:110,20:40] = 200
markers[90:110,90:110] = 100
ws = morphology.watershed(image_ext, markers)


# Plot
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5), sharex=True, sharey=True)

ax0.imshow(255-image_ext, cmap=plt.get_cmap('gray'))
ax0.set_title('Image')
ax0.axis('off')

ax1.imshow(markers, cmap=plt.get_cmap('gray'))
ax1.set_title('Markers')
ax1.axis('off')

ax2.imshow(ws, cmap=plt.get_cmap('nipy_spectral'), interpolation='nearest')
ax2.set_title('Watershed')
ax2.axis('off')

fig.tight_layout()
plt.show()