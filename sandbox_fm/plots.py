import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['toolbar'] = 'None'
class Visualization():
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        self.ax.axis('tight')
        self.ax.axis('off')
        # self.fig.canvas.toolbar.pack_forget()
        plt.ion()
        plt.show(block=False)
    def initialize(self, data):
        # create plots here
        pass
    def update(self, data):
        self.ax.clear()
        self.im = self.ax.imshow(
            data['kinect'],
            cmap='viridis_r',
            vmin=0.5,
            vmax=0.75
        )
        self.contour = self.ax.contour(data['kinect'])
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        self.ax.set_xlim(xlim[0] + 80, xlim[1] - 80)
        self.ax.set_ylim(ylim[0] + 80, ylim[1] - 80)
        self.ax.axis('tight')
        self.ax.axis('off')

        # Something like this would be faster
        # self.im.set_data(data['kinect'])
        # self.contour.set_data(data['kinect'])
        self.fig.canvas.draw()
        
        self.fig.canvas.flush_events()
