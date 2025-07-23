from matplotlib import pyplot as plt
import numpy as np
import cv2
from .curvature import compute_divided_curvature_radii


# 複数画像を表示する関数
def display_images(images, names=None, w=-1, h=-1):
    # plt.rcParams['font.size'] = 8
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Arial']

    fig = plt.figure(figsize=(30, 20))

    n = len(images)
    if names is None:
        names = [f'# {i + 1}' for i in range(n)]
    
    if w != -1:
        w = int(w)
        if n % w == 0:
            h = n // w
        else:
            h = int(n / w + 1)

    elif h != -1:
        h = int(h)
        if n % h == 0:
            w = n // h
        else:
            w = int(n / h + 1)
    
    else:
        w = n
        h = 1
    
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

""" 近似円の可視化: 円の半径の推移 """
def display_droplet_radii(list_arr_circle):
    n = len(list_arr_circle[0])
    plt.plot([arr_circle[0][2] for arr_circle in list_arr_circle], color='red', label='1st circle')
    if n > 1:
        plt.plot([arr_circle[1][2] for arr_circle in list_arr_circle], color='blue', label='2nd circle')
    if n > 2:
        plt.plot([arr_circle[2][2] for arr_circle in list_arr_circle], color='green', label='3rd circle')
    plt.yticks([0, 2 * list_arr_circle[0][0][2]])
    plt.legend()

""" 近似円の可視化: 元画像に円を加えた画像の列 """
def display_original_images_with_circles(instance, list_arr_circle, indices):   
    canvases = []
    names = []
    for i in indices:
        canvas = instance.gray_images[i].copy()
        cx, cy, cr = map(int, list_arr_circle[i][0])
        cv2.circle(canvas, (cx, cy), cr, 0, 1)
        cx, cy, cr = map(int, list_arr_circle[i][1])
        cv2.circle(canvas, (cx, cy), cr, 0, 3)
        canvases.append(canvas)
        names.append(f'image {i}')

    display_images(canvases, names, w=6)

""" 曲率半径の可視化: 特定のフレームについて、区画画像の列 """
def display_curvature_division_masks_of_a_frame(image_index, list_good_labels, list_divided_curvature_radii, list_division_masks):
    good_labels = list_good_labels[image_index].copy()
    curvature_radii = list_divided_curvature_radii[image_index]
    division_masks = list_division_masks[image_index].copy()
    images = np.array([good_labels] * len(division_masks)) + division_masks
    names = [f'Division {i}: {radius}' for i, radius in enumerate(curvature_radii)]
    display_images(images, names, w=4)

""" 曲率半径の可視化: 特定のフレームについて、区画ごとの正規化曲率半径 """
def display_curvature_division_radii_of_a_frame(image_index, list_arr_circle, list_divided_curvature_radii):
    cr = list_arr_circle[image_index][1][2]
    curvature_radii = list_divided_curvature_radii[image_index]
    xs = np.where(curvature_radii > 0)
    ys = (curvature_radii / cr)[xs]
    plt.scatter(xs, ys)
    plt.yticks([0, 1, 2])
    plt.xlabel('Division number')
    plt.ylabel('Normalized curvature radii')
    plt.legend()
    plt.show()


""" 曲率半径の可視化: 全フレームについて、特定区画の局率半径の列 """
def display_curvature_radii_of_a_division_in_frames(division_index, list_divided_curvature_radii, list_arr_circle):
    curvature_radii = np.array([divided_curvature_radii[division_index] for divided_curvature_radii in list_divided_curvature_radii])
    radii = np.array([arr_circle[1][2] for arr_circle in list_arr_circle])
    
    xs = np.where(curvature_radii > 0)
    ys = (curvature_radii / radii)[xs]
    plt.scatter(xs, ys)
    plt.plot(range(len(curvature_radii)), np.ones(len(curvature_radii)) * 0.9)
    plt.title(f'Normalized curvature radii of division {division_index}')
    plt.yticks([0, 1, 2])