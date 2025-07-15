import matplotlib.pyplot as plt
import cv2
import numpy as np


def get_extention_tag(f):
    return (f.split('.'))[-1]

# 複数画像を表示する関数
def display_images(images, names=None, w=-1):
    # plt.rcParams['font.size'] = 8
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Arial']

    fig = plt.figure(figsize=(30, 20))

    n = len(images)
    if names is None:
        names = [f'# {i + 1}' for i in range(n)]
    
    if w == -1:
        w = n
        h = 1
    else:
        w = int(w)
        if n % w == 0:
            h = n // w
        else:
            h = int(n / w + 1)
    
    for i in range(n):
        ax = fig.add_subplot(h, w, i + 1)
        ax.set_title(names[i], fontsize=20)
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(images[i])
    
    plt.show()

def display_stats(image):
    data = {
        'min': np.min(image),
        'max': np.max(image),
        'mean': np.mean(image),
        'std': np.std(image),
        'median': np.median(image)
    }
    print(data)