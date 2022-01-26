from typing import Union,List
import numpy
import cv2
import os

def load_image(image: Union[str, numpy.ndarray]):
    if isinstance(image, str):
        if not os.path.isfile(image):
            print("File {} does not exist!".format(image))
            return None
        return cv2.imread(image, 0)

    # Image alredy loaded
    elif isinstance(image, numpy.ndarray):
        return image

    # Format not recognized
    else:
        print("Unrecognized format: {}".format(type(image)))
        print("Unrecognized format: {}".format(image))
    return None

import matplotlib.pyplot as plt
def show_images(images: List[numpy.ndarray]) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.imshow(images[i])
    plt.show(block=True)

def main(dir='./hopper-none/'):
    im = []
    idx = 0
    for i in range(9):
        im.append(load_image(dir+'frame'+str(idx)+".jpg"))
        idx += 15
    show_images(im)

    

if __name__ == "__main__":
    main()
