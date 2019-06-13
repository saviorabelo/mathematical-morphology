# Import functions
from functions import *


file_name = './Dataset/13291732978427.jpg'
img = imread(file_name, as_grey=True)
img = (img * 255).round().astype(np.uint8)

bin1, media = threshold_mean(img) 
plots(img, bin1, media)

bin2, media_iter = threshold_mean_iter(img)
plots(img, bin2, media_iter)
