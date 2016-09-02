import matplotlib.pyplot as plt
import matplotlib.colors
import time

def depth_images(image):
    while True:
        try:
            img = plt.imread(image.name)
        except:
            time.sleep(0.001)
            continue
        depth = img[...,0]
        yield depth
        
