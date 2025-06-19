import numpy as np
import cv2
import skimage
from numba import jit
from .sensor_dust import create_sensor_dust_mask
from .grouping import divideIntoGroups
from .fitting import fit2ellipse
from .bright2dark import get_b2d_data
from .dense_area import get_dense_area


def to_binary(image):
    img = np.copy(image)
    img = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=0, sigmaY=0)
    img = cv2.Laplacian(img, -1, ksize=5)
    img = cv2.dilate(img, kernel=(5, 5), iterations=1)
    img = cv2.erode(img, kernel=(5, 5), iterations=1)
    _, binary = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    return binary

def get_noise_suppressed(binary):
    noise_suppressed = get_dense_area(binary, ksize=5, thresh_density=0.3)
    noise_suppressed = cv2.dilate(noise_suppressed, kernel=(5, 5), iterations=1)
    noise_suppressed = cv2.erode(noise_suppressed, kernel=(5, 5), iterations=1)
    return noise_suppressed

class Process:
    def __init__(self, path):
        self.gray_images = skimage.io.imread(path, plugin="tifffile")
        self.dust_mask = create_sensor_dust_mask(self.gray_images)
    
    # def run(self):
        # b2d_masks = []
        # ellipses = []
        # for gray in self.gray_images:
        #     binary = to_binary(gray)
        #     noise_suppressed = get_noise_suppressed(binary)

        #     sizes, grouped = divideIntoGroups(noise_suppressed, ksize=30)
        #     excluded = get_dust_excluded_image(noise_suppressed, grouped, self.dust_mask)
        #     median = cv2.medianBlur(excluded, ksize=7)
        #     e = fit2ellipse(median)
        #     cx, cy = e[0]

        #     b2d_vector_image, cos_image = get_b2d_data(gray, cx, cy, sobel_ksize=5)
        #     b2d_mask = 999999999999 # TODO
        #     masked = 999999999999 # TODO modify binary
        #     ellipse = fit2ellipse(masked)
        #     ellipses.append(ellipse)
        # self.ellipses = ellipses


# usage
# filenames = [f for f in os.listdir(base_dir) if get_extention_tag(f) == 'tif']
# paths = [base_dir + f for f in filenames]
# p = Process(paths)
    
