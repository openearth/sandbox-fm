import time
import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def depth_images(image):
    while True:
        try:
            img = plt.imread(image)
        except:
            logger.exception("can't read image %s", image)
            time.sleep(0.001)
            continue
        depth = img[..., 0]
        yield depth
