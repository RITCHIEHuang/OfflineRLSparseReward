import seaborn_image as isns
from typing import Union,List
import numpy as np
import numpy
import cv2
import os

from matplotlib import pyplot as plt


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


def load_images(dir):
    im = []
    idx = 0
    for i in range(9):
        image = load_image(dir+os.sep + 'frame' + str(idx) + ".jpg")
        image = np.flipud(image)
        im.append(image)
        idx += 100
    return im



if __name__ == '__main__':
    half_ie = load_images("./half-ie-video")
    half_none = load_images("./half-none-video")
    half_ia = load_images("./half-ia-video")
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np
    all_images = half_none + half_ie + half_ia

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 9),  # creates 2x2 grid of axes
                     axes_pad=0.05,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, all_images):
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.imshow(im)
    plt.show()
    plt.savefig('./halfcheetah_compare_frame.png')