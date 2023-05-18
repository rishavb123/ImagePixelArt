import numpy as np

from constants import *
from img_loader import Loader
from img_processor import Processor


class KMeans(Processor):
    def __init__(
        self,
        loader: Loader = None,
        k: int = 16,
        max_iterations: int = 10,
        convergence_criteria: float = 0,
        log: bool = False,
    ) -> None:
        super().__init__(loader)
        self.k = k
        self.max_iterations = max_iterations
        self.convergence_criteria = convergence_criteria
        self.log = log

        self.colors = None
        self.err = None

    def _process(self) -> np.ndarray:
        img = self.loader.bgr()
        img_flat = img.reshape(-1, 3)
        n_pixels = img_flat.shape[0]
        centroids = img_flat[np.random.choice(n_pixels, self.k, replace=False)]

        i = 0
        while i != self.max_iterations:
            i += 1

            distances = np.linalg.norm(
                img_flat[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
            )
            labels = np.argmin(distances, axis=1)
            old_centroids = np.copy(centroids)
            for j in range(self.k):
                centroids[j] = np.mean(img_flat[labels == j], axis=0)
            diff = np.linalg.norm(centroids - old_centroids)
            if self.log:
                print(f"Iteration {i} with diff {diff}")
            if diff <= self.convergence_criteria:
                break

        self.colors = centroids

        labels = np.reshape(labels, (img.shape[0], img.shape[1]))
        reduced_image = np.zeros_like(img)
        for i in range(self.k):
            reduced_image[labels == i] = centroids[i]

        return reduced_image


if __name__ == "__main__":
    loader = Loader.load(f"{DATA_PATH}/samples/img.jpg")
    alg = KMeans(loader, k=50, log=True, max_iterations=10)
    alg.save_processed_img(path=f"{DATA_PATH}/samples/kmeans_img.jpg")