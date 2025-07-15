import numpy as np
import cv2

def to_mask(image, mask_value):
    mask = np.zeros(image.shape, dtype='uint8')
    mask[image == mask_value] = 1
    return mask

def get_dilated_mask(mask, ksize=11):
    dilated_mask = np.copy(mask).astype('float32')
    dilated_mask = cv2.blur(dilated_mask, ksize=(ksize, ksize))
    dilated_mask[dilated_mask > 0] = 1
    return dilated_mask.astype('uint8')

def get_mask(image, thresh):
    dst = np.zeros(image.shape)
    dst[image >= thresh] = 1
    return dst