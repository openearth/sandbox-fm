import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import time


# Create sample data
y, x = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
nframes = 100
z={}
for ii in np.arange(0, nframes):
    z[ii] = np.sin(x-ii/(2*np.pi))*np.sin(x-ii/(2*np.pi))+np.sin(y)*np.sin(y)

print('Method 1: matplotlib, imshow')
fig = plt.figure(figsize=(16, 8), facecolor='white')
t_0 = time.process_time()
for ii in np.arange(0, nframes):
    plt.cla()
    plt.imshow(z[ii])
    plt.pause(0.001)
plt.close()
total_time = (time.process_time() - t_0)
frequency = nframes / total_time
print('Time taken: {}. A frequency of: {}'.format(total_time, frequency))


print('Method 2: matplotlib, imshow with set_data')
ii = 0
fig = plt.figure(figsize=(16, 8), facecolor='white')
plt.ion()
im = plt.imshow(z[ii])
t_0 = time.process_time()
for ii in np.arange(0, nframes):
    im.set_data(z[ii])
    plt.draw()
    plt.pause(0.001)
plt.close()
total_time = (time.process_time() - t_0)
frequency = nframes / total_time
print('Time taken: {}. A frequency of: {}'.format(total_time, frequency))

print('Method 3: matplotlib, pcolormesh (with set_array)')
ii = 0
fig = plt.figure(figsize=(16, 8), facecolor='white')
plt.ion()
quad = plt.pcolormesh(z[ii])
t_0 = time.process_time()
for ii in np.arange(0, nframes):
    quad.set_array(z[ii].ravel())
    plt.draw()
    plt.pause(0.001)
plt.close()
total_time = (time.process_time() - t_0)
frequency = nframes / total_time
print('Time taken: {}. A frequency of: {}'.format(total_time, frequency))


print('Method 3b: matplotlib, pcolormesh for larger window')
ii = 0
fig = plt.figure(figsize=(40, 30), facecolor='white')
plt.ion()
quad = plt.pcolormesh(z[ii])
t_0 = time.process_time()
for ii in np.arange(0, nframes):
    quad.set_array(z[ii].ravel())
    plt.draw()
    plt.pause(0.001)
plt.close()
total_time = (time.process_time() - t_0)
frequency = nframes / total_time
print('Time taken: {}. A frequency of: {}'.format(total_time, frequency))

from vispy import app
from vispy.gloo import clear, set_clear_color, set_viewport, Program

import vispy
# print(vispy.color.get_colormaps())


print('Method 4: vispy - app')


# Colormaps
colormaps = np.ones((16, 512, 4)).astype(np.float32)
values = np.linspace(0, 1, 512)[1:-1]

# Hot colormap
colormaps[0, 0] = 0, 0, 1, 1  # Low values  (< vmin)
colormaps[0, -1] = 0, 1, 0, 1  # High values (> vmax)
colormaps[0, 1:-1, 0] = np.interp(values, [0.00, 0.33, 0.66, 1.00],
                                          [0.00, 1.00, 1.00, 1.00])
colormaps[0, 1:-1, 1] = np.interp(values, [0.00, 0.33, 0.66, 1.00],
                                          [0.00, 0.00, 1.00, 1.00])
colormaps[0, 1:-1, 2] = np.interp(values, [0.00, 0.33, 0.66, 1.00],
                                          [0.00, 0.00, 0.00, 1.00])

# Grey colormap
colormaps[1, 0] = 0, 0, 1, 1  # Low values (< vmin)
colormaps[1, -1] = 0, 1, 0, 1  # High values (> vmax)
colormaps[1, 1:-1, 0] = np.interp(values, [0.00, 1.00],
                                          [0.00, 1.00])
colormaps[1, 1:-1, 1] = np.interp(values, [0.00, 1.00],
                                          [0.00, 1.00])
colormaps[1, 1:-1, 2] = np.interp(values, [0.00, 1.00],
                                          [0.00, 1.00])

img_vertex = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0 );
    v_texcoord = texcoord;
}
"""

img_fragment = """
uniform float vmin;
uniform float vmax;
uniform float cmap;
uniform sampler2D image;
uniform sampler2D colormaps;
uniform vec2 colormaps_shape;
varying vec2 v_texcoord;
void main()
{
    float value = texture2D(image, v_texcoord).r;
    float index = (cmap+0.5) / colormaps_shape.y;
    if( value < vmin ) {
        gl_FragColor = texture2D(colormaps, vec2(0.0,index));
    } else if( value > vmax ) {
        gl_FragColor = texture2D(colormaps, vec2(1.0,index));
    } else {
        value = (value-vmin)/(vmax-vmin);
        value = 1.0/512.0 + 510.0/512.0*value;
        gl_FragColor = texture2D(colormaps, vec2(value,index));
    }
}
"""
ii=0
class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(512, 512),
                            keys='interactive', always_on_top=True)
        self.image = Program(img_vertex, img_fragment, 4)
        self.image['position'] = (-1, -1), (-1, +1), (+1, -1), (+1, +1)
        self.image['texcoord'] = (0, 0), (0, +1), (+1, 0), (+1, +1)
        self.image['vmin'] = min(z[ii].ravel())
        self.image['vmax'] = max(z[ii].ravel())
        self.image['cmap'] = 0  # Colormap index to use

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.tick = 0

        self.image['colormaps'] = colormaps
        self.image['colormaps'].interpolation = 'linear'
        self.image['colormaps_shape'] = colormaps.shape[1], colormaps.shape[0]

        self.image['image'] = z[ii].astype('float32')
        self.image['image'].interpolation = 'linear'

        set_clear_color('black')

        self.show()

    def on_resize(self, event):
        width, height = event.physical_size
        set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        clear(color=True, depth=True)
        self.image.draw('triangle_strip')

    def on_timer(self, event):
        self.tick += 1
        if self.tick >= len(z):
            app.quit()
            return
        self.image['image'] = z[self.tick].astype('float32')
        self.update()


t_0 = time.process_time()
canvas = Canvas()
app.run()

total_time = (time.process_time() - t_0)
frequency = nframes / total_time
print('Time taken: {}. A frequency of: {}'.format(total_time, frequency))


# import vispy.scene
# import vispy.visuals

# print('Method 5: vispy - visual')
# ii = 0
# canvas = vispy.scene.SceneCanvas(keys='interactive')
# view = canvas.central_widget.add_view()
# # image = vispy.visuals.ImageVisual(z[ii])
# image = vispy.scene.Image(z[ii])

# view.add(image)
# view.camera = vispy.scene.TurntableCamera()
# view.camera.set_range((-20, 20), (-20, 20), (-20, 20))

# canvas.show()
# canvas.app.run()



# vispy.scene.visuals.Image(z[ii])
# vispy.scene.


# from pyqtgraph.Qt import QtCore, QtGui
# import pyqtgraph as pg

# app = QtGui.QApplication([])
# ii = 0
# win = pg.GraphicsLayoutWidget()
# win.show()  ## show widget alone in its own window
# win.setWindowTitle('pyqtgraph example: ImageItem')
# view = win.addViewBox()
# view.setAspectLocked(True)

# img = pg.ImageItem(border='w')

# view.addItem(img)
# view.setRange(QtCore.QRectF(0, 0, 100, 100))
# # data = np.random.normal(size=(15, 600, 600), loc=1024, scale=64).astype(np.uint16)

# # img.setImage(z[ii])
# # import os
# # outputdir = r'D:\scripting\sandbox-fm\playground'

# img.setImage(z[ii])
# QtGui.QApplication.instance().exec_()


# for ii in np.arange(0, nframes):
#     img.setImage(z[ii])
# #     img.save(os.path.join(outputdir,'test_{}'.format(ii)))

# # img.save(os.path.join(outputdir,'test_{}'.format(ii)))