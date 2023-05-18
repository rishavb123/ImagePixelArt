import numpy as np
import cv2

from constants import *

picks_dir = "picks"

alpha = 0.5
beta = 0.5

n_x = 32
n_y = 18

c = 0

colors = cv2.imread(f"{DATA_PATH}/samples/img_processed_upscaled.jpg")

img = np.zeros((400 * n_y, 400 * n_x, 3))

for i in range(n_y):
    for j in range(n_x):

        with open(f"{DATA_PATH}/{picks_dir}/{c}_{i}_{j}.txt") as f:
            img_name = f.read()

        photo = cv2.imread(f"{DATA_PATH}/{DATASET_NAME}/{img_name}")
        color = colors[i * 400 : i * 400 + 400, j * 400 : j * 400 + 400]

        img[i * 400 : i * 400 + 400, j * 400 : j * 400 + 400] = cv2.addWeighted(photo, alpha, color, beta, 0.0)

        c += 1

cv2.imwrite(f"{picks_dir}_result.jpg", img)