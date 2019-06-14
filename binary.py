# -*- coding: utf-8 -*-
# Import library
import numpy as np
from skimage.io import imread
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import disk


file_name = './Dataset/13291732978427.jpg'
folha = imread(file_name)
folha = rgb2gray(folha)

folha = folha.max() - folha
radios = 5
dilat = morphology.dilation(folha, disk(radios))
eros  = morphology.erosion(folha, disk(radios))

""" fig = plt.figure(figsize=(10,10))
a = fig.add_subplot(1,4,1)
plt.imshow(folha, cmap=plt.get_cmap('gray'))
a.set_title('Original')
plt.axis('off')

a = fig.add_subplot(1,4,2)
plt.imshow(dilat, cmap=plt.get_cmap('gray'))
a.set_title('Dilatação')
plt.axis('off')

a = fig.add_subplot(1,4,3)
plt.imshow(eros, cmap=plt.get_cmap('gray'))
a.set_title('Erosão')
plt.axis('off')

a = fig.add_subplot(1,4,4)
plt.imshow(disk(radios), cmap=plt.get_cmap('gray'))
a.set_title('E. Estrut.')
plt.axis('off')

plt.tight_layout()
plt.show()   """  


cont_ext = morphology.dilation(folha, disk(radios)) - folha
cont_int = folha - morphology.erosion(folha, disk(radios))
cont = morphology.dilation(folha, disk(radios)) - morphology.erosion(folha, disk(radios))

""" fig = plt.figure(figsize=(100,100))
a = fig.add_subplot(1,3,1)
plt.imshow(cont_ext, cmap=plt.get_cmap('gray'))
a.set_title('Contorno Ext.')
plt.axis('off')

a = fig.add_subplot(1,3,2)
plt.imshow(cont_int, cmap=plt.get_cmap('gray'))
a.set_title('Contorno Int.')
plt.axis('off')

a = fig.add_subplot(1,3,3)
plt.imshow(cont, cmap=plt.get_cmap('gray'))
a.set_title('Contorno ')
plt.axis('off')

plt.tight_layout()
plt.show() """

eros_3 = morphology.erosion(folha, disk(3)) 
eros_15 = morphology.erosion(folha, disk(15)) 
eros_30 = morphology.erosion(folha, disk(30)) 


""" fig = plt.figure(figsize=(100,100))
a = fig.add_subplot(1,3,1)
plt.imshow(eros_3, cmap=plt.get_cmap('gray'))
a.set_title('Erosão (3)')
plt.axis('off')

a = fig.add_subplot(1,3,2)
plt.imshow(eros_15, cmap=plt.get_cmap('gray'))
a.set_title('Erosão (15)')
plt.axis('off')

a = fig.add_subplot(1,3,3)
plt.imshow(eros_30, cmap=plt.get_cmap('gray'))
a.set_title('Erosão (30)')
plt.axis('off')

plt.tight_layout()
plt.show() """


open_3  = morphology.opening(folha, disk(3))
open_15 = morphology.opening(folha, disk(15))
open_30 = morphology.opening(folha, disk(30))

""" fig = plt.figure(figsize=(100,100))
a = fig.add_subplot(1,4,1)
plt.imshow(folha, cmap=plt.get_cmap('gray'))
a.set_title('Original')
plt.axis('off')

a = fig.add_subplot(1,4,2)
plt.imshow(open_3, cmap=plt.get_cmap('gray'))
a.set_title('Abert (3)')
plt.axis('off')

a = fig.add_subplot(1,4,3)
plt.imshow(open_15, cmap=plt.get_cmap('gray'))
a.set_title('Abert (15)')
plt.axis('off')

a = fig.add_subplot(1,4,4)
plt.imshow(open_30, cmap=plt.get_cmap('gray'))
a.set_title('Abert (30)')
plt.axis('off')

plt.tight_layout()
plt.show()  """


dif_3  = folha - open_3
dif_15 = folha - open_15
dif_30 = folha - open_30

""" fig = plt.figure(figsize=(100,100))
a = fig.add_subplot(1,4,1)
plt.imshow(dif_3, cmap=plt.get_cmap('gray'))
a.set_title('Original')
plt.axis('off')

a = fig.add_subplot(1,4,2)
plt.imshow(dif_15, cmap=plt.get_cmap('gray'))
a.set_title('Abert (3)')
plt.axis('off')

a = fig.add_subplot(1,4,3)
plt.imshow(dif_30, cmap=plt.get_cmap('gray'))
a.set_title('Abert (15)')
plt.axis('off')

plt.tight_layout()
plt.show()  """


dif_open = folha - morphology.opening(folha, disk(radios))

""" fig = plt.figure(figsize=(100,100))
a = fig.add_subplot(1,2,1)
plt.imshow(folha, cmap=plt.get_cmap('gray'))
a.set_title('Original')
plt.axis('off')

a = fig.add_subplot(1,2,2)
plt.imshow(dif_open, cmap=plt.get_cmap('gray'))
a.set_title('Opening')
plt.axis('off')

plt.tight_layout()
plt.show()  """