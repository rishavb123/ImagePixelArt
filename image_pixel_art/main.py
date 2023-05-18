from constants import *
from img_loader import Loader
from img_processor import CompositeProcessor
from pixelize import Pixelize
from kmeans import KMeans

def main():
    img_path = f"{DATA_PATH}/samples/img.jpg"

    loader = Loader.load(img_path)
    alg = CompositeProcessor(
        loader,
        Pixelize(pixel_size=50),
        KMeans(k=25, log=True, max_iterations=10),
    )
    alg.save_processed_img()

if __name__ == "__main__":
    main()