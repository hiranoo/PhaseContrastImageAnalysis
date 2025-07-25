import cv2
import numpy as np
import csv
from ruamel.yaml import YAML


def cut_noise(image, dust_mask=None, sigma=2):
    dst = image.copy()
    if dust_mask is not None:
        dst[dust_mask == 1] = 0
    thresh = np.mean(dst) + np.std(dst) * sigma
    dst[dst < thresh] = 0
    return dst

def compute_mass_center(image, dust_mask, sigma):
    msk = cut_noise(image, dust_mask, sigma)
    m = cv2.moments(msk, binaryImage=True)
    cx, cy= m['m10']/m['m00'] , m['m01']/m['m00']
    return cx, cy

def get_extention_tag(f):
    return (f.split('.'))[-1]

def save_as_csv(arr, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(arr)
        
def load_yaml(path):
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(path, 'r') as f:
        data = yaml.load(f)
    return data

def save_yaml(data, path):
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(path, 'w') as f:
        yaml.dump(data, f)