# -*- coding: utf-8 -*-
# Import library
import numpy as np
import matplotlib.pyplot as plt

def show_images_ws(image, dilation, markers, watershed):
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

def threshold_mean(img):
    media = np.mean(img)

    img_bin = ((img > media) * 255).astype("uint8")
    return img_bin, media

def threshold_mean_iter(img):
    mean_initial = np.mean(img)
    while 1:
        mean_1 = np.mean(img[img < mean_initial])        
        mean_2 = np.mean(img[img >= mean_initial])
        
        media_new = (mean_1 + mean_2) / 2
        if media_new.astype("uint8") == mean_initial.astype("uint8"):
            break
        else:
            mean_initial = media_new

    img_bin = ((img > media_new) * 255).astype("uint8")
    return img_bin, media_new

def plots(p1, p2, p3):
    fig = plt.figure(figsize=(9,3), dpi=80)
    a = fig.add_subplot(1,3,1)
    a.axis('off')
    plt.imshow(p1, cmap=plt.get_cmap('gray'))
    a.set_title('Original')

    a = fig.add_subplot(1,3,2)
    a.axis('off')
    plt.imshow(p2, cmap=plt.get_cmap('gray'))
    a.set_title('Limiarização')

    a = fig.add_subplot(1,3,3)
    a.hist(p1.ravel(), bins=128, normed=True, edgecolor='none', histtype='stepfilled')
    a.axvline(x=p3,color='orange')
    plt.tight_layout()

    plt.show()

def plot_hist(img):
    hist_args = {'title':'Histogram', 
                'xlabel':'Gray',
                'ylabel':'Pixels',
                'xticks':[0, 32, 64, 96, 128, 160, 192, 224, 256]}

    fig = plt.figure(figsize=(9,4.5), dpi=80)

    a = fig.add_subplot(1,2,1)
    a.imshow(img, cmap=plt.get_cmap('gray'))
    a.axis('off')
    a.set_title('Original')

    a = fig.add_subplot(1,2,2, **hist_args)
    a.hist(img.ravel(), bins=128, normed=True, edgecolor='none',histtype='stepfilled')
    plt.tight_layout()
    plt.show()