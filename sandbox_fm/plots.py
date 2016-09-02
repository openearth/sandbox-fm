import matplotlib.pyplot as plt


class Visualization():
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        self.ax.axis('tight')
        self.ax.axis('off')
        plt.show(block=False)

    def update(self, data):
        self.ax.imshow(data['kinect'], cmap='gist_earth')
        self.ax.autoscale()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
