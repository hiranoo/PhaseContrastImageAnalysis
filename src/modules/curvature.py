import numpy as np
import cv2
from numba import jit
from .process import *
from .fitting import fit2circle_points
from .labels import count_label_sizes


""" 近似円を扇形に分割した円弧に対応する点群ごとの曲率半径を集計 """

@jit(nopython=True)
def create_divided_image(image, cx, cy, standard_vec, angle):
    dst = np.zeros(image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] == 0:
                continue
            vec = np.array([x - cx, y - cy])
            r = np.linalg.norm(vec)
            if np.dot(vec / r, standard_vec) >= np.cos(angle):
                dst[y, x] = 1
    return dst

@jit(nopython=True)
def find_double_arcs(mask, cx, cy, labels):
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            label_id = labels[y, x]
            if label_id == 0:
                continue
            v = np.array([x - cx, y - cy])
            v /= np.linalg.norm(v)
            for y2 in range(mask.shape[0]):
                for x2 in range(mask.shape[1]):
                    label_id2 = labels[y2, x2]
                    if label_id2 == 0 or label_id == label_id2:
                        continue
                    v2 = np.array([x2 - cx, y2 - cy])
                    v2 /= np.linalg.norm(v2)
                    if (1 - np.dot(v, v2)) < 1e-6:
                        return 1
    return 0

def remove_noise_areas(mask, cx, cy, thresh):
    nlabels, labels = cv2.connectedComponents(mask.astype('uint8'))
    dst = mask.copy()
    if find_double_arcs(mask, cx, cy, labels) == 1:
        # ラベルサイズの比が大きければ、最大のラベルのみ抽出する
        label_sizes = count_label_sizes(nlabels, labels)
        max_label_ratio = np.max(label_sizes[1:]) / np.sum(label_sizes[1:])
        if max_label_ratio >= thresh:
            max_label_id = int(np.argmax(label_sizes))
            dst[labels != max_label_id] = 0
        else:
            dst = np.zeros(dst.shape)
    return dst

def compute_divided_curvature_radii(good_labels, cx, cy, division_number, angle):
    masks = []
    for standard_angle in np.linspace(0, 2 * np.pi, division_number + 1)[:-1]:
        standard_vec = np.array([np.cos(standard_angle), np.sin(standard_angle)])
        mask = create_divided_image(good_labels, cx, cy, standard_vec, angle)
        mask = remove_noise_areas(mask, cx, cy, thresh=0.75)
        masks.append(mask)
    
    curvature_radii = []
    for mask in masks:
        if np.all(mask == 0):
            radius = -1
        else:
            ys, xs = np.where(mask == 1)
            points = [(x, y) for x, y in zip(xs, ys)]
            radius = fit2circle_points(points)[2]
        curvature_radii.append(radius)
    
    return np.array(curvature_radii), np.array(masks)

def compute_list_divided_curvature_radii(list_good_labels, list_arr_circle, division_number, angle):
    list_divided_curvature_radii = []
    list_division_masks = []
    for good_labels, arr_circle in zip(list_good_labels, list_arr_circle):
        cx, cy, cr = arr_circle[1]
        divided_curvature_radii, masks = compute_divided_curvature_radii(good_labels, cx, cy, division_number, angle)
        list_divided_curvature_radii.append(divided_curvature_radii)
        list_division_masks.append(masks)
    return np.array(list_divided_curvature_radii), np.array(list_division_masks)