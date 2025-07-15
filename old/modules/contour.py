import numpy as np
import cv2
from .mask import to_mask, get_dilated_mask
from .fitting import fit2circle


""" 1. 基準となる輪郭の選定 """
# 明るさの微分（b2d_vector size）が十分大きい画素を最も多く含む連続領域を計算
def _compute_largest_brightness_diff_label_id(b2d_vector_image, dust_mask, nlabels, labels):
    large_diff_mask = b2d_vector_image.copy()
    large_diff_mask[get_dilated_mask(dust_mask) == 1] = 0
    large_diff_mask[100:150, 140:200] = 0
    large_diff_mask[large_diff_mask < np.mean(b2d_vector_image) + np.std(b2d_vector_image)] = 0
    large_diff_mask[large_diff_mask > 0] = 1
    large_diff_mask = large_diff_mask.astype('uint8')

    labels[large_diff_mask == 0] = 0
    count_masked_labels = np.zeros(nlabels)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label_id = labels[y, x]
            if label_id == 0:
                continue
            count_masked_labels[label_id] += 1

    return np.argmax(count_masked_labels)

# 明るさの微分が十分大きい画素を最も多く含む連続領域　に対応する輪郭番号を返す
def get_largest_brightness_diff_contour_id(contours, edge_binary, b2d_vector_image, dust_mask):
    nlabels, labels = cv2.connectedComponents(edge_binary.astype('uint8'))
    largest_brightness_diff_label_id = _compute_largest_brightness_diff_label_id(b2d_vector_image, dust_mask, nlabels, labels)
    label_mask = to_mask(labels, largest_brightness_diff_label_id)

    counts = np.zeros(len(contours))
    for i, contour in enumerate(contours):
        for element in contour:
            x, y = element[0]
            if label_mask[y, x] == 1:
                counts[i] += 1
    return np.argmax(counts)


""" 2. 基準の輪郭と同心円状にある輪郭列の選定 """
def _is_inside_circle(x, y, cx, cy, cr):
    return (x - cx)**2 + (y - cy)**2 <= cr**2

def _do_cross_contour_and_circle(contour, cx, cy, cr, width=2):
    for p in contour:
        x, y = p[0]
        if _is_inside_circle(x, y, cx, cy, cr + width / 2) and not _is_inside_circle(x, y, cx, cy, cr - width/2):
            return True
    return False

def select_concentric_contour_indices(contours, standard_contour_id, circle_thickness=10):
    x, y, r = fit2circle([contours[standard_contour_id]])
    concentric_contour_indices = []
    for i, contour in enumerate(contours):
        if _do_cross_contour_and_circle(contour, x, y, r, circle_thickness):
            concentric_contour_indices.append(i)
    return concentric_contour_indices