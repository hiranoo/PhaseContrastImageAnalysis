import numpy as np
from numba import jit


@jit(nopython=True)
def validate(x, begin_x, end_x):
    return begin_x <= x and x < end_x

# binary imageについて，高密度領域を抽出する
@jit(nopython=True)
def extractDenseArea(image, ksize, thresh):
    dst = np.zeros(image.shape, dtype='uint8')
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            cnt = 0
            for dy in range(-ksize//2, ksize//2 + 1):
                if (not validate(y + dy, 0, image.shape[0])):
                    continue
                for dx in range(-ksize//2, ksize//2 + 1):
                    if (not validate(x + dx, 0, image.shape[1])):
                        continue
                    if (image[y + dy, x + dx] > 0):
                        cnt += 1
            density = cnt / (ksize * ksize)
            if (density >= thresh):
                dst[y, x] = 1
    return dst

def get_dense_area(image, ksize, thresh_density):
    assert(0 <= thresh_density and thresh_density <= 1)
    return extractDenseArea(image, ksize, thresh_density)