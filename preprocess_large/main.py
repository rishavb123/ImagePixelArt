from constants import *
from img_loader import Loader
from img_processor import CompositeProcessor
from pixelize import Pixelize
from kmeans import KMeans

def main():
    img_path = f"{DATA_PATH}/samples/img.png"

    loader = Loader.load(img_path)

    pixelize = Pixelize(pixel_size=50)
    # kmeans = KMeans(k=25, log=True, max_iterations=10)

    alg = CompositeProcessor(
        loader,
        pixelize,
        # kmeans,
    )
    alg.save_processed_img()
    # print(alg.calc_error(), pixelize.pixel_num)

if __name__ == "__main__":
    main()