import cv2
import numpy as np
from tqdm import tqdm
import os

from constants import *

imgs_path = f"{DATA_PATH}/{DATASET_NAME}"

imgs = []
img_names = os.listdir(imgs_path)

print("Loading in images")

for img_name in tqdm(img_names):
    imgs.append(cv2.imread(f"{imgs_path}/{img_name}"))

print("Converting to numpy")

imgs = np.array(imgs)
np.save(f"{DATA_PATH}/processed_imgs.npy", imgs)
np.save(f"{DATA_PATH}/processed_img_names.npy", img_names)
print(imgs.shape)