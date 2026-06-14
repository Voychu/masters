import matplotlib.pyplot as plt
import numpy

def plot_image(img):
    fig, ax = plt.subplots(1,1)
    ax.imshow(img)
    ax.set_title("visualize")
    plt.show()