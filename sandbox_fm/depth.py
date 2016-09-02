import matplotlib.pyplot as plt
import matplotlib.colors

def read_depth_image(filename):
    rgb = plt.imread(filename)
    depth = rgb[...,0]
    return depth
