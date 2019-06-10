# Import library
import matplotlib.pyplot as plt

def show_images(image, markers, watershed):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5), sharex=True, sharey=True)

    ax0.imshow(image, cmap=plt.get_cmap('gray'))
    ax0.set_title('Image')
    ax0.axis('off')

    ax1.imshow(markers, cmap=plt.get_cmap('gray'))
    ax1.set_title('Markers')
    ax1.axis('off')

    ax2.imshow(watershed, cmap=plt.get_cmap('nipy_spectral'), interpolation='nearest')
    ax2.set_title('Watershed')
    ax2.axis('off')

    fig.tight_layout()
    plt.show()