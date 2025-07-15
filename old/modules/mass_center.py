import numpy as np
import cv2
from .edge import get_bright_edge
from .basic_process import cut_noise

def compute_mass_center(gray, dilated_dust_mask, flat_image=None):
    if flat_image is None:
        flat_image = cv2.GaussianBlur(gray, ksize=(301, 301), sigmaX=0, sigmaY=0)
    flat_laplacian = abs(cv2.Laplacian(flat_image, cv2.CV_32F, ksize=5))

    edge = get_bright_edge(gray)
    edge[dilated_dust_mask == 1] = 0

    m = cv2.moments(edge, binaryImage=True)
    cx, cy= m['m10']/m['m00'] , m['m01']/m['m00']
    return cx, cy


def compute_mass_center2(image, dust_mask, sigma):
    msk = cut_noise(image, dust_mask, sigma)
    m = cv2.moments(msk, binaryImage=True)
    cx, cy= m['m10']/m['m00'] , m['m01']/m['m00']
    return cx, cy