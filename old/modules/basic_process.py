import numpy as np
import os
import cv2

def cut_noise(image, dust_mask=None, sigma=2):
    dst = image.copy()
    if dust_mask is not None:
        dst[dust_mask == 1] = 0
    thresh = np.mean(dst) + np.std(dst) * sigma
    dst[dst < thresh] = 0
    return dst