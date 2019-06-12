# Import library
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import morphology
from skimage.morphology import disk, square, rectangle
import matplotlib.pyplot as plt

def show_images(image, dilation, markers, watershed):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, figsize=(8, 2.5), sharex=True, sharey=True)

    ax0.imshow(image)
    ax0.set_title('Original')
    ax0.axis('off')

    ax1.imshow(dilation, cmap=plt.get_cmap('gray'))
    ax1.set_title('External Dilation')
    ax1.axis('off')

    ax2.imshow(markers, cmap=plt.get_cmap('gray'))
    ax2.set_title('Markers')
    ax2.axis('off')
    
    ax3.imshow(watershed, cmap=plt.get_cmap('nipy_spectral'), interpolation='nearest')
    ax3.set_title('Watershed')
    ax3.axis('off')

    fig.tight_layout()
    plt.show()
# End