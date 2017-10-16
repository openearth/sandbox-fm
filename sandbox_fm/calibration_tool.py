import pathlib
import json
import logging
import pathlib

import cv2
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt


from .calibrate import (
    compute_transforms
)
from .sandbox_fm import (
    update_delft3d_initial_vars
)

from .depth import (
    calibrated_height_images,
)
from .calibrate import (
    transform,
    compute_transforms
)

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class PolygonInteractor(object):
    """
    An polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly, markevery=None, pstate=True):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly
        self.pstate = pstate
        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True, markevery=markevery)
        if self.pstate:
            textc = 'black'
            self.TL = self.ax.annotate('TL', xy=(x[0], y[0]), xytext=(x[0] + 1, y[0] + 1),
                                       animated=True, color=textc)
            self.TR = self.ax.annotate('TR', xy=(x[1], y[1]), xytext=(x[1] + 1, y[1] + 1),
                                       animated=True, color=textc)
            self.BL = self.ax.annotate('BL', xy=(x[2], y[2]), xytext=(x[2] + 1, y[2] + 1),
                                       animated=True, color=textc)
            self.BR = self.ax.annotate('BR', xy=(x[3], y[3]), xytext=(x[3] + 1, y[3] + 1),
                                       animated=True, color=textc)

        self.ax.add_line(self.line)
        #self._update_line(poly)

        cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

        self.canvas = canvas

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        if self.pstate:
            self.ax.draw_artist(self.TL)
            self.ax.draw_artist(self.TR)
            self.ax.draw_artist(self.BL)
            self.ax.draw_artist(self.BR)
        self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = [tup for i, tup in enumerate(self.poly.xy) if i != ind]
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.array(
                        list(self.poly.xy[:i]) +
                        [(event.xdata, event.ydata)] +
                        list(self.poly.xy[i:]))
                    self.line.set_data(zip(*self.poly.xy))
                    break

        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        self.line.set_data(zip(*self.poly.xy))
        self.canvas.restore_region(self.background)
        if self.pstate:
            if self._ind == 0:
                self.TL.xy = (x, y)
                self.TL.xytext = (x + 1, y + 1)
                self.ax.draw_artist(self.TL)
            if self._ind == 1:
                self.TR.xy = (x, y)
                self.TR.xytext = (x + 1, y + 1)
                self.ax.draw_artist(self.TR)
            if self._ind == 2:
                self.BL.xy = (x, y)
                self.BL.xytext = (x + 1, y + 1)
                self.ax.draw_artist(self.BL)
            if self._ind == 3:
                self.BR.xy = (x, y)
                self.BR.xytext = (x + 1, y + 1)
                self.ax.draw_artist(self.BR)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.draw()

class Calibration(object):
    """
    key-bindings:

    'esc' Use escape button to close figure when done calibrating

    'enter' Use enter key to save changes to calibration.json

    """

    def __init__(self, path, videos, raws, model):
        # absolute path to calibration.json
        self.path = path.absolute()
        self.videos = videos
        self.raws = raws
        self.model = model
        self.data = {}
        self.height_points = []
        self.model_points = []
        self.img_points = []
        self.z_values = []
        # define fixed box coordinate system (what will be on the screen)
        # 0 at bottom
        self.box = np.array([
            [0, 0],
            [640, 0],
            [640, 480],
            [0, 480]
        ], dtype='float32')

        # save the current working directory
        self.curdir = pathlib.Path.cwd()
        self.make_window()
        # get data from model
        update_delft3d_initial_vars(self.data, self.model)
        if self.path.exists():
            with open(str(self.path)) as f:
                self.old_calibration = json.load(f)
        else:
            self.old_calibration = {}

    def make_window(self):
        self.fig, self.axes = plt.subplots(2, 3)
        self.fig.figsize = (200, 100)
        # sic show instructions in the title
        self.fig.suptitle('select 4 points (clockwise start at top left)')
        self.axes[0, 0].set_title("1) projector box")
        self.axes[0, 1].set_title("2) model box")
        self.axes[1, 0].set_title("3) set low (red dot), high")
        self.axes[1, 1].set_title("4) result with raw kinect image")
        self.axes[1, 2].set_title("5) initial view of sandbox")

    def save(self):
        with open(str(self.path), 'w') as f:
            json.dump(self.result, f, indent=2, cls=NumpyEncoder)

    @property
    def z(self):
        zk = self.model.get_var('zk')
        return zk.min(), zk.max()

    @property
    def result(self):
        model_points = self.model_points
        height_points = self.height_points
        img_points = self.img_points
        box = self.box
        z_values = self.z_values


        comment = """
        This file contains calibrations for model %s.
        It is generated with the perspective transform from opencv.
        """ % (self.model.configfile, )
        result = {
            "box": box,
            "img_points": img_points,
            "model_points": model_points,
            "height_points": height_points,
            "z_values": z_values,
            "z": self.z,
            "_comment": comment

        }
        return result

    def show_result(self, ax):
        # we should have results by now
        # save the calibration info
        data = self.data
        result = self.result

        # add the transforms
        result.update(compute_transforms(result))

        xy_node = np.c_[
            data['xk'],
            data['yk'],
            np.ones_like(data['xk'])
        ].astype('float32')

        print(result['model2box'], xy_node)
        # now for showing results
        xy_nodes_in_img = np.squeeze(
            cv2.perspectiveTransform(
                np.dstack([
                    xy_node[:, np.newaxis, 0],
                    xy_node[:, np.newaxis, 1]
                ]).astype('float32'),
                np.array(result['model2box'], dtype='float32')
            )
        )
        # scatter plot
        ax.scatter(
            xy_nodes_in_img[:, 0],
            xy_nodes_in_img[:, 1],
            c=data['zk'].ravel(),
            cmap='Greens',
            edgecolor='none',
            s=10,
            alpha=0.2
        )

        # transformed video on top
        '''ax.imshow(
            cv2.warpPerspective(
                next(self.videos),
                np.array(result['img2box'], dtype='float32'),
                (640, 480)
            ),
            cmap='Reds',
            alpha=0.5
        )'''

        # Still raw image, with cut out by polygon
        ax.imshow(
            cv2.warpPerspective(
                next(self.raws),
                np.array(result['img2box'], dtype='float32'),
                (640, 480)
            ),
            cmap='jet',
        )
        return result

    def show_data(self, ax, result):
        print('wat het moet zijn: ')
        print("([933, 923], [-15.65193178225327, 7.227255704551149], PosixPath('/home/sandbox/src/sandbox-fm/tests/zandmotor/anomaly.npy'))")

        print(result['z_values'],result['z'], pathlib.Path('anomaly.npy').absolute())
        heights = calibrated_height_images(
            result['z_values'],
            result['z'],
            anomaly_name=pathlib.Path('anomaly.npy').absolute()
        )
        height = next(heights)
        warped_height = cv2.warpPerspective(
            height.filled(0),
            np.array(result['img2box']),
            (640, 480)
        )
        self.im_height = ax.imshow(
            warped_height,
            'jet',
            #cmap=terrajet2,
            alpha=1,
            vmin=result['z'][0],
            vmax=result['z'][-1],
            visible=True
        )
        plt.title("5) Deze plot doet nog niet precies wat ik wil")
        plt.show()

    def add_edit_polygon(self, ax, points=4):
        xmin, xmax = ax.get_xlim()
        dx = xmax - xmin
        xmin += 0.1 * dx
        xmax -= 0.1 * dx
        ymin, ymax = ax.get_ylim()
        dy = ymax - ymin
        ymin += 0.1 * dy
        ymax -= 0.1 * dy
        if isinstance(points, int):
            if points == 4:
                # same order as self.box
                xs = [xmin, xmax, xmax, xmin]
                ys = [ymin, ymin, ymax, ymax]
            elif points == 2:
                xs = [xmin, xmax]
                ys = [ymin, ymax]
        elif isinstance(points, list):
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
        else:
            raise ValueError('I only understand 2 or 4 points')

        poly = Polygon(
            list(zip(xs, ys)),
            animated=True,
            closed=False,
            facecolor='red',
            alpha=0.3
        )
        ax.add_patch(poly)
        markevery = None
        pstate = True
        if len(xs) == 2:
            markevery = [1]
            pstate = False
        p = PolygonInteractor(ax, poly, markevery=markevery, pstate=pstate)
        return p

    def run(self):

        fig, axes = self.fig, self.axes
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        # get video and depth image
        video = next(self.videos)
        raw = next(self.raws)

        # show the depth and video in the left window

        axes[0, 0].imshow(raw) #, cmap='jet', vmin=650, vmax=800)
        axes[1, 0].imshow(raw)

        # convert to array we can feed into opencv
        data = self.data
        axes[0, 1].scatter(
            data['xk'].ravel(),
            data['yk'].ravel(),
            c=data['zk'].ravel(),
            cmap='viridis',
            edgecolor='none'
        )

        img_points = self.old_calibration.get("img_points", 4)
        img_poly = self.add_edit_polygon(axes[0, 0], points=img_points)
        model_points = self.old_calibration.get("model_points", 4)
        model_poly = self.add_edit_polygon(axes[0, 1], points=model_points)
        height_points = self.old_calibration.get("height_points", 2)
        height_poly = self.add_edit_polygon(axes[1, 0], points=2)

        self.axes[0, 2].text(0, 0.7, "1) shows the raw kinect image, use the dots to select the area to \n" +
                                    "(TL = Top Left, TR = Top Right, BL = Bottom Left, BR = Bottom Right)\n" +
                                    "2) Shows the domain to select within the bathymetry of the model \n" +
                                    "3) Select the lowest (red dot) and highest point in the raw kinect data\n"
                                    "4) This box shows the result \n" +
                                    "5) Shows the initial calibrated bathymetry as will be displayed")
        self.axes[0, 2].axis('off')
        # keep track of the selected points\n

        height_points = self.height_points

        z_values = self.z_values


        # pointer event
        pid = None

        # define the point selector
        def picker(event):
            if event.key == 'enter':
                # stop listening we're done
                self.img_points = list(zip(
                    *img_poly.line.get_data()
                ))
                self.model_points = list(zip(
                    *model_poly.line.get_data()
                ))
                self.height_points = list(zip(
                    *height_poly.line.get_data()
                ))
                u0, v0 = (
                    int(np.round(self.height_points[0][0])),
                    int(np.round(self.height_points[0][1]))
                )
                u1, v1 = (
                    int(np.round(self.height_points[1][0])),
                    int(np.round(self.height_points[1][1]))
                )
                self.z_values = [
                    raw[v0, u0],
                    raw[v1, u1]
                ]
                # deep should be > undeep
                if not self.z_values[0] > self.z_values[1]:
                    logger.warn("I had to reverse your z points, please select lower value with the bottom left dot. Coordinates: %s, raw: %s.",
                                self.height_points,
                                self.z_values )
                    self.z_values = list(reversed(self.z_values))
                    self.height_points = list(reversed(self.height_points))
                self.save()
                result = self.show_result(axes[1, 1])
                self.show_data(axes[1, 2], result)

            if event.key =='escape':
                fig.canvas.mpl_disconnect(pid)
                plt.close(fig)

        plt.ion()
        pid = fig.canvas.mpl_connect('key_press_event', picker)
        plt.show(block=True)
