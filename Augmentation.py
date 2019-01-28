from PIL import Image
import glob
import cv2
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

image_crop_size = 112

def crop(img):
    width, height = img.size  # Get dimensions

    left = (width - image_crop_size) / 2
    top = (height - image_crop_size) / 2
    right = (width + image_crop_size) / 2
    bottom = (height + image_crop_size) / 2

    return img.crop((left, top, right, bottom))


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



array_img = np.array(img)
manip_img = gamma_correction(array_img, 0.8)

manip_img = jpg_compression(manip_img, 70)
manip_img = resizing(manip_img, 2.0)

manip_img = Image.fromarray(manip_img)

manip_img = crop(manip_img)
plt.imshow(manip_img)
plt.show()