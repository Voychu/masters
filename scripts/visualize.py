import matplotlib.pyplot as plt
import numpy as np

def plot_image(img):
    fig, ax = plt.subplots(1,1)
    ax.imshow(img)
    ax.set_title("visualize")
    plt.show()

def plot_hist(data_column,title):
    fig,ax = plt.subplots(1,1)
    u,c = np.unique(data_column, return_counts=True)
    density = c / np.sum(c)
    ax.bar(u, density, edgecolor='black', width=0.8)
    ax.set_xticks(u)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()