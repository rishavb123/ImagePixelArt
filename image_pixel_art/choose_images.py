import cv2
import numpy as np

from constants import *

start_num = 351
show_num = 10

imgs = np.load(f"{DATA_PATH}/processed_imgs.npy")
mu = np.mean(imgs, axis=(1, 2)).astype("uint8")
img_names = np.load(f"{DATA_PATH}/processed_img_names.npy")


def pick(i, j, c, idx):
    with open(f"{DATA_PATH}/picks/{c}_{i}_{j}.txt", "w") as f:
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

for i in range(h):
    for j in range(w):
        if c < start_num:
            c += 1
            continue

        color = big_img[i, j]

        color_img = np.zeros((400, 400 * show_num, 3), np.uint8)
        color_img[:] = color
        put_text(color_img, f"{c} / {h * w}")

        dists = np.linalg.norm(mu - color, axis=1)
        idx = np.argpartition(dists, show_num)[:show_num]

        chosen_imgs = imgs[idx]

        for k in range(show_num):
            put_text(chosen_imgs[k], f"{k}: {dists[idx[k]]:0.4f}")

        merged = np.concatenate(
            (color_img, np.concatenate(chosen_imgs, axis=1)), axis=0
        )

        merged = cv2.resize(merged, (200 * show_num, 400))

        cv2.imshow("Color Picker", merged)

        inp = cv2.waitKey(0)

        print(c, inp, color, i, j)

        if inp == ord("q"):
            done = True
        else:
            pick(i, j, c, idx[inp - 48])

        c += 1

        if done:
            break

    if done:
        break

cv2.destroyAllWindows()