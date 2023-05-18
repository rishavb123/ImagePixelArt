import numpy as np
from typing import Callable

from constants import *
from img_loader import Loader
from img_processor import Processor


class Pixelize(Processor):
    def __init__(
        self, loader: Loader = None, combiner: Callable = np.mean, pixel_size: int = 50
    ) -> None:
        super().__init__(loader)
        self.combiner = combiner
        self.pixel_size = pixel_size

    def _process(self) -> np.ndarray:
        img = np.copy(self.loader.bgr())
        h, w, _ = img.shape

        for i in range(0, h, self.pixel_size):
            for j in range(0, w, self.pixel_size):
                img[i : i + self.pixel_size, j : j + self.pixel_size] = self.combiner(
                    img[i : i + self.pixel_size, j : j + self.pixel_size].reshape(
                        -1, 3
                    ),
                    axis=0,
                )

        return img


if __name__ == "__main__":
    loader = Loader.load(f"{DATA_PATH}/samples/img.jpg")
    alg = Pixelize(loader, pixel_size=10)
    alg.save_processed_img(path=f"{DATA_PATH}/samples/pixel_img.jpg")