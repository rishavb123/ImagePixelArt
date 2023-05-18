import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np

DATA_PATH = "C:/Data/ImagePixelData"
NEW_SIGMA = 10

raw_path = f"{DATA_PATH}/processed"
processed_path = f"{DATA_PATH}/processed2_{NEW_SIGMA}"

raw_imgs = os.listdir(raw_path)

if os.path.exists(processed_path):
    shutil.rmtree(processed_path)

os.makedirs(processed_path)

for img_name in tqdm(raw_imgs):
    try:
        img = cv2.imread(f"{raw_path}/{img_name}")
        img_flat = img.reshape(-1, 3)
        mu = np.mean(img_flat, axis=0)
        sigma = np.std(img_flat, axis=0)

        z = img - mu
        z = z / np.max(sigma)

        new_img = z * NEW_SIGMA + mu

        cv2.imwrite(f"{processed_path}/{img_name}", new_img)
    except:
        import sys
        print(img_name)
        sys.exit()