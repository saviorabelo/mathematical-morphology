# -*- coding: utf-8 -*-
# Import library
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.color import rgb2gray
from skimage.io import imread

# Convert to float: Important for subtraction later which won't work with uint8
#image = img_as_float(data.coins())
#image = gaussian_filter(image, 1)

image = rgb2gray(imread('Dataset/13291658032668.jpg'))
image = image.max() - image

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image

dilated = reconstruction(seed, mask, method='dilation')

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1,
                                    ncols=4,
                                    figsize=(8, 2.5),
                                    sharex=True,
                                    sharey=True)

ax0.imshow(image, cmap='gray')
ax0.set_title('original image')
ax0.axis('off')

ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
ax1.set_title('dilated')
ax1.axis('off')

ax2.imshow(image - dilated, cmap='gray')
ax2.set_title('image - dilated')
ax2.axis('off')

ax3.imshow(seed, cmap='gray')
ax3.set_title('seed')
ax3.axis('off')

fig.tight_layout()


h = 0.2
seed = image - h
dilated = reconstruction(seed, mask, method='dilation')
hdome = image - dilated

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, figsize=(8, 2.5))
yslice = 200

ax0.plot(mask[yslice], '0.5', label='mask')
ax0.plot(seed[yslice], 'k', label='seed')
ax0.plot(dilated[yslice], 'r', label='dilated')
ax0.set_ylim(-0.2, 2)
ax0.set_title('image slice')
ax0.set_xticks([])
ax0.legend()

ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
ax1.axhline(yslice, color='r', alpha=0.4)
ax1.set_title('dilated')
ax1.axis('off')

ax2.imshow(hdome, cmap='gray')
ax2.axhline(yslice, color='r', alpha=0.4)
ax2.set_title('image - dilated')
ax2.axis('off')

ax3.imshow(seed, cmap='gray')
ax3.axhline(yslice, color='r', alpha=0.4)
ax3.set_title('seed')
ax3.axis('off')

fig.tight_layout()
plt.show()