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

def plot_age_hist_bins(data_column):
    fig,ax = plt.subplots(1,1)
    u,c = np.unique(data_column, return_counts=True)
    x = [0,0,0,0,0]
    for i in range(len(u)):
        if u[i] > 82:
            x[4]+=c[i]
        elif u[i] >=75:
            x[3]+=c[i]
        elif u[i] >=65:
            x[2]+=c[i]
        elif u[i] >=55:
            x[1]+=c[i]
        elif u[i] >=45:
            x[0]+=c[i]
        else:
            continue
    
    print(x)
    xs = ['45-54','55-64','65-74','75-84','>85']
    ax.bar(xs, x, edgecolor='black', width=0.8)
    ax.set_ylim(bottom=0)
    ax.set_title("histogram wieku")
    plt.tight_layout()
    plt.show()