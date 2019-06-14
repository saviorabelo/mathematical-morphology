import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction

# Convert to float: Important for subtraction later which won't work with uint8
image = img_as_float(data.coins())
image = gaussian_filter(image, 1)

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image

dilated = reconstruction(seed, mask, method='dilation')

# Exibe imagens
fig = plt.figure(figsize=(20,20))
a = fig.add_subplot(1,3,1)
plt.imshow(image, cmap=plt.cm.gray)
a.set_title('Máscara')
plt.axis('off')

a = fig.add_subplot(1,3,2)
plt.imshow(seed, cmap=plt.cm.gray)
a.set_title('Marcador')
plt.axis('off')

a = fig.add_subplot(1,3,3)
plt.imshow(dilated, cmap=plt.cm.gray)
a.set_title('Reconstrução')
plt.axis('off')


