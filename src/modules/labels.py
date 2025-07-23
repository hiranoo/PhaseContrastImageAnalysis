import numpy as np
from matplotlib import pyplot as plt
from numba import jit

@jit(nopython=True)
def _count_label_sizes(nlabels, labels):
    res = np.zeros(nlabels)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label_id = labels[y, x]
            if label_id > 0:
                res[label_id] += 1
    return res

def count_label_sizes(nlabels, labels):
    return _count_label_sizes(nlabels, labels)

def get_large_label_indices(nlabels, labels, thresh_size, thresh_sigma):
    label_sizes = count_label_sizes(nlabels, labels)
    thresh = min(thresh_size, np.mean(label_sizes) + np.std(label_sizes) * thresh_sigma)
    return np.where(label_sizes > thresh)[0]

def display_label_sizes(nlabels, labels, thresh_size, thresh_sigma):
    label_sizes = count_label_sizes(nlabels, labels)
    plt.scatter(range(len(label_sizes)), label_sizes) # debug
    plt.plot([np.mean(label_sizes) + np.std(label_sizes)] * len(label_sizes), label=f'mean + {thresh_sigma} sigma')
    plt.plot(np.ones(len(label_sizes)) * thresh_size, label=f'{thresh_size}')
    plt.xlabel('label id')
    plt.ylabel('label size')
    plt.legend()
    plt.show()