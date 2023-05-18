import os
import shutil
import cv2
from tqdm import tqdm

DATA_PATH = "C:/Data/ImagePixelData"

raw_path = f"{DATA_PATH}/raw"
processed_path = f"{DATA_PATH}/processed"

raw_imgs = os.listdir(raw_path)

if os.path.exists(processed_path):
    shutil.rmtree(processed_path)

os.makedirs(processed_path)

for img_name in tqdm(raw_imgs):
    try:
        img = cv2.imread(f"{raw_path}/{img_name}")
        h, w, _ = img.shape
        if h > w:
            diff = h - w
            img = img[diff // 2:-diff // 2, :]
        elif w > h:
            diff = w - h
            img = img[:, diff // 2:-diff // 2]
        img = cv2.resize(img, (400, 400))
        cv2.imwrite(f"{processed_path}/{img_name}", img)
    except:
        import sys
        print(img_name)
        sys.exit()