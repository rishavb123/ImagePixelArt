import abc
import cv2
import numpy as np

from img_loader import Loader


class Processor(abc.ABC):
    def __init__(self, loader: Loader = None) -> None:
        self.loader = loader

        self.processed_img = None
        self.err = None

    @abc.abstractmethod
    def _process(self) -> np.ndarray:
        pass

    def process(self):
        if self.processed_img is None:
            self.err = None
            self.processed_img = self._process()
        return self.processed_img

    def save_processed_img(self, path: str = None):
        self.process()
        if path is None:
            splt_lst = self.loader.path.split(".")
            ext = splt_lst[-1]
            rem = ".".join(splt_lst[:-1])
            path = f"{rem}_processed.{ext}"
        cv2.imwrite(path, self.processed_img)

    def show_images(self):
        self.process()
        cv2.imshow("Original Image", self.loader.bgr())
        cv2.imshow("Processed Image", self.processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calc_error(self):
        if self.err is not None:
            return self.err
        self.process()
        self.err = np.linalg.norm(self.loader.bgr() - self.processed_img)
        return self.err


class CompositeProcessor(Processor):
    def __init__(self, loader: Loader, *processors: Processor) -> None:
        super().__init__(loader)
        self.processors = processors

    def _process(self) -> np.ndarray:
        cur_loader = self.loader()
        for processor in self.processors:
            processor.loader = cur_loader
            cur_loader = Loader(img=processor._process())
        return cur_loader.bgr()
