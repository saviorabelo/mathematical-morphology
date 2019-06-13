# Import functions
from functions import *

file_name = './Dataset/13291732978427.jpg'
img = imread(file_name)
image = rgb2gray(img)
image_ext = morphology.dilation(image, disk(5)) - image

# Watershed
m,n = image.shape
markers = np.zeros([m,n])
m = math.floor(m/2)
n = math.floor(n/2)
markers[20:40,20:40] = 200
markers[m:m+20,n:n+20] = 100 # Center
ws = morphology.watershed(image_ext, markers)

# Plot
show_images_ws(img, 255-image_ext, markers, ws)
