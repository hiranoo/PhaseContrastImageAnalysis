import numpy as np
import cv2
import skimage
from numba import jit
from modules.display import display_images
from modules.fitting import fit2circle_points, fit_points_to_circle_iteration, cross_circle_and_labels
from modules.mask import get_dilated_mask, get_mask
from modules.bright2dark import create_b2d_data, reduce_b2d_noise
from modules.labels import get_large_label_indices, display_label_sizes

class MyImages:
    def __init__(self, path):
        self.gray_images = skimage.io.imread(path, plugin="tifffile")
        self.dust_mask = np.zeros(self.gray_images[0].shape)

def test_run(instance: MyImages, index: int, conf):
    images = []
    gray = np.copy(instance.gray_images[index])
    blurred = cv2.GaussianBlur(gray, ksize=(conf['gaussian_blur_ksize'], conf['gaussian_blur_ksize']), sigmaX=0, sigmaY=0)
    dust_mask = get_dilated_mask(instance.dust_mask, conf['dust_mask_dilate_ksize'])
    b2d_vector_image, b2d_cos_image, b2d_mass_center = create_b2d_data(blurred, dust_mask, conf['sobel_ksize'], conf['b2d_vector_mass_center_sigma'])
    b2d_cos_mask = get_mask(b2d_cos_image, conf['b2d_cos_thresh'])
    noise_reduced_b2d_vector_image = reduce_b2d_noise(b2d_vector_image, b2d_cos_mask, dust_mask, conf['b2d_vector_blur_ksize'], conf['b2d_vector_noise_sigma'])
    
    # 十分大きい連続領域を検出する
    nlabels, labels = cv2.connectedComponents(noise_reduced_b2d_vector_image.astype('uint8'))
    display_label_sizes(nlabels, labels, conf['large_label_thresh_size'], conf['large_label_thresh_sigma'])
    large_label_indices = get_large_label_indices(nlabels, labels, conf['large_label_thresh_size'], conf['large_label_thresh_sigma'])
    large_labels = np.zeros(labels.shape)
    for index in large_label_indices:
        large_labels[labels == index] = 1

    # 近似円の内側の余計な連続領域を排除する
    arr_circle, _ = fit_points_to_circle_iteration(large_labels, itr=1)
    good_label_indices = large_label_indices[cross_circle_and_labels(labels, large_label_indices, arr_circle[0][0], arr_circle[0][1], arr_circle[0][2], w_in=conf['cross_extra_width_inside'], w_out=conf['cross_extra_width_outside'], thresh=conf['cross_ratio_thresh'])]
    
    # 残った連続領域上の点群を円近似
    good_labels = np.zeros(labels.shape)
    for label_id in good_label_indices:
        good_labels[labels == label_id] = 1
    arr_circle, arr_points = fit_points_to_circle_iteration(good_labels, w_out=conf['fitting_extra_width_outside'], itr=conf['fitting_iteraion_number'])

    # draw
    canvas = np.copy(gray)
    for i in range(len(arr_circle)):
        cx, cy, cr = map(int, arr_circle[i])
        cv2.circle(canvas, (cx, cy), cr, color=0, thickness=1+i)

    print('mass center of b2d vectors', b2d_mass_center)
    images = [gray, blurred, b2d_vector_image, b2d_cos_image, b2d_cos_mask, noise_reduced_b2d_vector_image, labels, large_labels, good_labels, canvas]
    names = ['gray', 'blurred', 'b2d_vector_image', 'b2d_cos_image', 'b2d_cos_mask', 'noise reduced b2d_vector', 'labels', 'large_labels', 'good labels', 'circle fitting']
    display_images(images, names, w=4)
    return images

def run(instance, indices, conf):
    list_good_labels = []
    list_arr_circle = []
    list_arr_points = []
    dust_mask = get_dilated_mask(instance.dust_mask)
    for index in indices:
        gray = instance.gray_images[index]
        blurred = cv2.GaussianBlur(gray, ksize=(conf['gaussian_blur_ksize'], conf['gaussian_blur_ksize']), sigmaX=0, sigmaY=0)
        dust_mask = get_dilated_mask(instance.dust_mask, conf['dust_mask_dilate_ksize'])
        b2d_vector_image, b2d_cos_image, _ = create_b2d_data(blurred, dust_mask, conf['sobel_ksize'], conf['b2d_vector_mass_center_sigma'])
        b2d_cos_mask = get_mask(b2d_cos_image, conf['b2d_cos_thresh'])
        noise_reduced_b2d_vector_image = reduce_b2d_noise(b2d_vector_image, b2d_cos_mask, dust_mask, conf['b2d_vector_blur_ksize'], conf['b2d_vector_noise_sigma'])
        
        # 十分大きい連続領域を検出する
        nlabels, labels = cv2.connectedComponents(noise_reduced_b2d_vector_image.astype('uint8'))
        large_label_indices = get_large_label_indices(nlabels, labels, conf['large_label_thresh_size'], conf['large_label_thresh_sigma'])
        large_labels = np.zeros(labels.shape)
        for index in large_label_indices:
            large_labels[labels == index] = 1

        # 近似円の内側の余計な連続領域を排除する
        arr_circle, _ = fit_points_to_circle_iteration(large_labels, itr=1)
        good_label_indices = large_label_indices[cross_circle_and_labels(labels, large_label_indices, arr_circle[0][0], arr_circle[0][1], arr_circle[0][2], w_in=conf['cross_extra_width_inside'], w_out=conf['cross_extra_width_outside'], thresh=conf['cross_ratio_thresh'])]
        
        # 残った連続領域上の点群を円近似
        good_labels = np.zeros(labels.shape)
        for label_id in good_label_indices:
            good_labels[labels == label_id] = 1
        arr_circle, arr_points = fit_points_to_circle_iteration(good_labels, w_out=conf['fitting_extra_width_outside'], itr=conf['fitting_iteraion_number'])

        list_good_labels.append(good_labels)
        list_arr_circle.append(arr_circle)
        list_arr_points.append(arr_points)
    
    return list_good_labels, list_arr_circle, list_arr_points