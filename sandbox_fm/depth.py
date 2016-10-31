import time
import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def depth_images(image, skip=3):
    """generate depth images, skipping skip number of cells"""
    while True:
        try:
            img = plt.imread(image)
        except:
            logger.exception("can't read image %s", image)
            time.sleep(0.001)
            continue
        depth = img[::skip, ::skip, 0]
        yield depth
