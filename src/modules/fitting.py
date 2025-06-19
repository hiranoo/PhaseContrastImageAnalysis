import numpy as np
import cv2
from numba import jit
from .grouping import divideIntoGroups
from .sensor_dust import exclude_dust_groups


def fit2ellipse(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    valid_contour = np.concatenate([c for c in contours if len(c) >= 5])
    return cv2.fitEllipse(valid_contour)

def get_fitted_ellipse(image, dust_mask):
    sizes, grouped = divideIntoGroups(image, ksize=30)
    excluded = exclude_dust_groups(image, grouped, dust_mask)
    median = cv2.medianBlur(excluded, ksize=7)
    return fit2ellipse(median)

# algorithm: https://www.eyedeal.co.jp/img/eyemLib_fitting1.pdf
def _fit2circle(contours):
    A = np.concatenate([np.array([[p[0][0], p[0][1], 1] for p in contour]) for contour in contours])
    v = np.sum(np.array(np.square(A.T[:2].T)), axis=1)

    #QR分解
    Q, R = np.linalg.qr(A)

    # x = ( 2*x_0, 2*y_0, r^2-x0^2-y0^2 )
    x = np.linalg.inv(R)@Q.T@v

    #解
    x_0 = x[0]/2 #円の中心座標
    y_0 = x[1]/2 #円の中心座標
    radius = np.sqrt(x[2] + x_0**2 + y_0**2)

    return x_0, y_0, radius

def _fit2circle_points(points):
    A = np.array([np.array([p[0], p[1], 1]) for p in points])
    v = np.sum(np.array(np.square(A.T[:2].T)), axis=1)

    #QR分解
    Q, R = np.linalg.qr(A)

    # x = ( 2*x_0, 2*y_0, r^2-x0^2-y0^2 )
    x = np.linalg.inv(R)@Q.T@v

    #解
    x_0 = x[0]/2 #円の中心座標
    y_0 = x[1]/2 #円の中心座標
    radius = np.sqrt(x[2] + x_0**2 + y_0**2)

    return x_0, y_0, radius

def fit2circle(contours):
    return _fit2circle(contours)


def fit2circle_points(points):
    return _fit2circle_points(points)

@jit(nopython=True)
def _is_inside_circle(x, y, cx, cy, cr):
    return (x - cx)**2 + (y - cy)**2 <= cr**2

@jit(nopython=True)
def _get_points_inside_points_circle(binary_image, cx, cy, cr, w_out):
    points = []
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 1 and _is_inside_circle(x, y, cx, cy, cr + w_out):
                points.append((x, y))
    return points

def fit_points_to_circle_iteration(binary_image, cx=0, cy=0, cr=1e6, w_out=0, itr=2):
    arr_points  = []
    arr_circle = []
    for i in range(itr):
        points = _get_points_inside_points_circle(binary_image, cx, cy, cr, w_out)
        cx, cy, cr = fit2circle_points(points)
        arr_points.append(points)
        arr_circle.append((cx, cy, cr))
    return arr_circle, arr_points

@jit(nopython=True)
def _cross_circle_and_labels(labels, label_indices, cx, cy, cr, w_in, w_out, thresh):
    total = {}
    cross = {}
    for label_id in label_indices:
        total[label_id] = cross[label_id] = 0
    
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label_id = labels[y, x]
            if label_id in label_indices:
                total[label_id] += 1
                if _is_inside_circle(x, y, cx, cy, cr + w_out) and not _is_inside_circle(x, y, cx, cy, cr - w_in):
                    cross[label_id] += 1
    
    cross_ratios = np.array([cross[label_id] / total[label_id] for label_id in label_indices])
    return cross_ratios > thresh

def cross_circle_and_labels(labels, label_indices, cx, cy, cr, w_in=5, w_out=50, thresh=0):
    return _cross_circle_and_labels(labels, label_indices, cx, cy, cr, w_in, w_out, thresh)