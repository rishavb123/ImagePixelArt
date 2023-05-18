from __future__ import annotations

import cv2
import numpy as np


class Loader:

    loaders = {}

    @staticmethod
    def load(path: str) -> Loader:
        if path not in Loader.loaders:
            Loader.loaders[path] = Loader(path)
        return Loader.loaders[path]

    def __init__(self, path: str = None, img: np.ndarray = None) -> None:
        self.path = path
        self.bgr_image = img
        self.rgb_image = None
        self.gray_image = None

    def bgr(self) -> np.ndarray:
        if self.bgr_image is None:
            self.bgr_image = cv2.imread(self.path)
        return self.bgr_image

    def rgb(self) -> np.ndarray:
        self.bgr()
        self.rgb_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2RGB)

    def gray(self) -> np.ndarray:
        self.bgr()
        self.gray_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)
