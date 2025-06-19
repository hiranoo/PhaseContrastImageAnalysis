import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from numba import jit
from .grouping import divideIntoGroups


@jit(nopython=True)
def validate(x, begin_x, end_x):
    return begin_x <= x and x < end_x

@jit(nopython=True)
def get_dust_group_indices(grouped, dust_mask):
    dust_group_indices = set()
    for y in range(grouped.shape[0]):
        for x in range(grouped.shape[1]):
            group_id = grouped[y, x]
            if group_id >= 0 and dust_mask[y, x] == 1:
                dust_group_indices.add(group_id)
    return list(dust_group_indices)

@jit(nopython=True)
def remove_dust_groups(img, grouped, dust_group_indices):
    dst = img.copy()
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            group_id = grouped[y, x]
            if group_id in dust_group_indices:
                dst[y, x] = 0
    return dst

def create_sensor_dust_mask(source_images):
    sum_images = np.zeros(source_images[0].shape, dtype='float32')
    for img in source_images:
        gauss = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0, sigmaY=0)
        sum_images += gauss
    sum_images -= np.min(sum_images)
    lap = cv2.Laplacian(sum_images, -1, ksize=5)

    blur = cv2.GaussianBlur(lap, ksize=(7, 7), sigmaX=0, sigmaY=0)

    # edges: 0, other points: 1
    binary = np.ones(blur.shape, dtype=int)
    binary[abs(blur) < np.mean(abs(blur)) + np.std(abs(blur)) * 3] = 0

    sizes, grouped = divideIntoGroups(binary, ksize=5)

    candidates = [group_id for group_id in range(len(sizes)) if sizes[group_id] >= 50 and sizes[group_id] < 1000]
    points = {}
    for gi in candidates:
        points[gi] = []
    for y in range(0, grouped.shape[0]):
        for x in range(0, grouped.shape[1]):
            group_id = int(grouped[y, x])
            if group_id in candidates:
                points[group_id].append((x, y))

    dusts = []
    for gi in candidates:
        xs = np.array([p[0] for p in points[gi]])
        ys = np.array([p[1] for p in points[gi]])
        cov = np.cov(np.array([xs, ys]))
        if abs(cov[0, 0] / cov[1, 1] - 1) < 2:
            dusts.append(gi)

    mask_dust = np.zeros(grouped.shape, dtype='uint8')
    for y in range(0, grouped.shape[0]):
        for x in range(0, grouped.shape[1]):
            group_id = int(grouped[y, x])
            if group_id in dusts:
                mask_dust[y, x] = 1
    
    return mask_dust

def exclude_dust_groups(img, grouped, dust_mask):
    dust_group_indices = get_dust_group_indices(grouped, dust_mask)
    dst = remove_dust_groups(img, grouped, dust_group_indices)
    return dst
