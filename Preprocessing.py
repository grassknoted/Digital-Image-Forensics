import os
import itertools
import re
import sys
from pathlib import Path
import multiprocessing as mp
from PIL import Image
import cv2
import numpy as np
from random import shuffle
from random import * 
import matplotlib.pyplot as plt
from io import BytesIO
import argparse
import glob
import numpy as np
import pandas as pd
from imageio import imread
from os.path import join
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, downscale_local_mean
import skimage.exposure
import scipy.ndimage
import skimage
from tqdm import tqdm
from io import BytesIO
import scipy.misc
import jpeg4py as jpeg
from scipy import signal
import math
import csv
from sklearn.utils import class_weight
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import  islice
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread
from subprocess import check_output

# Cropped Image Dimensions
image_crop_size = 112

# Number of images from each class
image_limit = 4

# Number of images to manipulate form each class
image_manips = 2

# Image sizes
size = (224, 224)

# Possible manipulations
MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']

def gamma_correction(array_img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(array_img, table)


def jpg_compression(array, quality):
    img = Image.fromarray(array)
    img.save('img.jpg', "JPEG", quality=quality)
    return cv2.cvtColor(cv2.imread('img.jpg'), cv2.COLOR_BGR2RGB)


def resizing(array_img, factor):
    h, w, ch = array_img.shape
    return cv2.resize(array_img, (int(factor * w), int(factor * h)), interpolation=cv2.INTER_CUBIC)

def get_crop(img, crop_size, random_crop=False):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop = crop_size // 2
    pad_x = max(0, crop_size - img.shape[1])
    pad_y = max(0, crop_size - img.shape[0])
    if (pad_x > 0) or (pad_y > 0):
        img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2), (0,0)), mode='wrap')
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    if random_crop:
        freedom_x, freedom_y = img.shape[1] - crop_size, img.shape[0] - crop_size
        if freedom_x > 0:
            center_x += np.random.randint(math.ceil(-freedom_x/2), freedom_x - math.floor(freedom_x/2) )
        if freedom_y > 0:
            center_y += np.random.randint(math.ceil(-freedom_y/2), freedom_y - math.floor(freedom_y/2) )

    return img[center_y - half_crop : center_y + crop_size - half_crop, center_x - half_crop : center_x + crop_size - half_crop]

print(check_output(["ls", "./Data/"]).decode("utf8"))

input_path = Path('./Data/')
train_path = input_path / 'train'
test_path = input_path / 'test'

cameras = os.listdir(train_path)

train_images = []
train_labels = []

for camera in cameras:
    image_count = 0
    for fname in sorted(os.listdir(train_path / camera)):
        if(image_count<image_limit):
            print(camera, fname)
            img = Image.open(train_path / camera / fname)
            #img = img.resize(size, Image.BICUBIC) 
            #img = img.thumbnail(size, Image.ANTIALIAS)
            array_img = np.array(img)
            train_images.append(array_img)
            train_labels.append(camera)
            #train_camera.append(fname)
            image_count += 1


img = train_images[19]

array_img = img

manip_img = get_crop(array_img, image_crop_size)
plt.imshow(img)
plt.show()

plt.imshow(manip_img)
plt.show()

print("Initial Length: ", len(train_images))

for j in range(10):
    for manip in MANIPULATIONS:
        for i in range(image_manips):
            to_consider = randint((j*image_limit), (image_limit*(j+1))-1)
            img = train_images[to_consider]
            current_label = train_labels[to_consider]
            array_img = img
            img = get_crop(array_img, image_crop_size)
            if manip.startswith('jpg'):
                quality = int(manip[3:])
                out = BytesIO()
                im = Image.fromarray(img)
                im.save(out, format='jpeg', quality=quality)
                im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
                del out
            
            elif manip.startswith('gamma'):
                gamma = float(manip[5:])
                im_decoded = np.uint8(cv2.pow(img / 255., gamma)*255.)

            elif manip.startswith('bicubic'):
                scale = float(manip[7:])
                im_decoded = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            
            else:
                assert False

            save_path = './Data/New/'+current_label+'/'+str(manip)+'manip_'+str(i)+'.jpg'
            scipy.misc.imsave(save_path, im_decoded)
            print("New Image added: ", save_path)
            train_images.append(im_decoded)
            train_labels.append(current_label)

print("Final Length: ", len(train_images))