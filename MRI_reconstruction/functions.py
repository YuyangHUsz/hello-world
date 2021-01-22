import torch
from matplotlib import pyplot as plt


def show_slice(data, cmap=None):
    plt.imshow(data, cmap=cmap)
    plt.show()
    plt.axis('off')


