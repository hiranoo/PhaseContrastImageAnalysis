import numpy as np
import os
import cv2
from numba import jit
from .mass_center import compute_mass_center2
from .basic_process import cut_noise


@jit(nopython=True)
def jit_norm(vec):
    return np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])

@jit(nopython=True)
def jit_cross(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

@jit(nopython=True)
def jit_equal(a, b):
    eps = 1e-6
    return abs(a - b) < eps

@jit(nopython=True)
def compute_b2d_data(cx, cy, v_diff, h_diff):
    height = v_diff.shape[0]
    width = v_diff.shape[1]
    bright2dark_vector_image = np.zeros(v_diff.shape)
    cosin_image = np.zeros(v_diff.shape)
    for y in range(height):
        for x in range(width):
            vec1 = np.array([x - cx, y - cy])
            vec2 = np.array([-h_diff[y, x], -v_diff[y, x]])
            norm1 = norm(vec1)
            norm2 = norm(vec2)
            bright2dark_vector_image[y, x] = norm2
            if (equal(norm1, 0) or equal(norm2, 0)):
                continue
            cos = cross(vec1, vec2) / (norm1 * norm2)
            cosin_image[y, x] = cos
    return bright2dark_vector_image, cosin_image

def get_b2d_data(image, cx, cy, sobel_ksize):
    sobel_vertical = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    sobel_horizontal = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    return compute_b2d_data(cx, cy, sobel_vertical, sobel_horizontal)

#####################################################################################

@jit(nopython=True)
def _create_b2d_vector_image(v_diff, h_diff):
    height = v_diff.shape[0]
    width = v_diff.shape[1]
    bright2dark_vector_image = np.zeros(v_diff.shape)
    for y in range(height):
        for x in range(width):
            vec = np.array([-h_diff[y, x], -v_diff[y, x]])
            norm = jit_norm(vec)
            bright2dark_vector_image[y, x] = norm
    return bright2dark_vector_image

@jit(nopython=True)
def _create_b2d_cos_image(cx, cy, v_diff, h_diff):
    height = v_diff.shape[0]
    width = v_diff.shape[1]
    b2d_cos_image = np.zeros(v_diff.shape)
    for y in range(height):
        for x in range(width):
            vec1 = np.array([x - cx, y - cy])
            vec2 = np.array([-h_diff[y, x], -v_diff[y, x]])
            norm1, norm2 = jit_norm(vec1), jit_norm(vec2)
            if (jit_equal(norm1, 0) or jit_equal(norm2, 0)):
                continue
            b2d_cos_image[y, x] = jit_cross(vec1, vec2) / (norm1 * norm2)
    return b2d_cos_image

def create_b2d_data(image, dust_mask, sobel_ksize, sigma):
    sobel_vertical = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    sobel_horizontal = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    b2d_vector_image = _create_b2d_vector_image(sobel_vertical, sobel_horizontal)
    cx, cy = compute_mass_center2(b2d_vector_image, dust_mask, sigma)
    b2d_cos_image = _create_b2d_cos_image(cx, cy, sobel_vertical, sobel_horizontal)
    return b2d_vector_image, b2d_cos_image, (cx, cy)

def reduce_b2d_noise(b2d_vector_image, b2d_cos_mask, dust_mask, ksize, sigma):
    dst = np.copy(b2d_vector_image)
    dst[dust_mask == 1] = 0
    dst[b2d_cos_mask == 0] = 0
    dst = cv2.GaussianBlur(dst, (ksize, ksize), sigmaX=0, sigmaY=0)
    dst = cut_noise(dst, sigma=sigma)
    return dst