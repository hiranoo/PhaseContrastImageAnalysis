import numpy as np
import cv2


def get_bright_edge(gray):
    blurred = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=0, sigmaY=0)
    lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=5)
    dilated = cv2.dilate(lap, kernel=(5, 5), iterations=1)
    eroded = cv2.erode(dilated, kernel=(5, 5), iterations=1)
    dst = -eroded
    dst[dst < 0] = 0
    return dst


def get_dark_edge(gray):
    blurred = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=0, sigmaY=0)
    lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=5)
    dilated = cv2.dilate(lap, kernel=(5, 5), iterations=1)
    dst = cv2.erode(dilated, kernel=(5, 5), iterations=1)
    dst[dst < 0] = 0
    return dst