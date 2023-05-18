import cv2
import numpy as np

from constants import *

start_num = 0
show_num = 10

imgs = np.load(f"{DATA_PATH}/processed_imgs.npy")
mu = np.mean(imgs, axis=(1, 2)).astype("uint8")
img_names = np.load(f"{DATA_PATH}/processed_img_names.npy")


def pick(i, j, c, idx):
    with open(f"{DATA_PATH}/random_picks/{c}_{i}_{j}.txt", "w") as f:
        f.write(img_names[idx])

def put_text(img, txt):
    font = cv2.FONT_HERSHEY_SIMPLEX
    textPos = (10, 50)
    fontScale = 1
    fontColor = (0, 0, 0)
    thickness = 3
    lineType = 2

    cv2.putText(
        img,
        txt,
        textPos,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )


big_img = cv2.imread(f"{DATA_PATH}/samples/img_processed_downscaled.jpg")

h, w, _ = big_img.shape

c = 0

done = False

random_picks = np.random.choice(imgs.shape[0], size=(h * w), replace=False)

for i in range(h):
    for j in range(w):
        if c < start_num:
            c += 1
            continue

        pick(i, j, c, random_picks[c])

        c += 1

        if done:
            break

    if done:
        break

cv2.destroyAllWindows()