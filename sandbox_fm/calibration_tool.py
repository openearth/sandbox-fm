import pathlib
import json
import logging
import pathlib

import cv2
import numpy as np
import matplotlib
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
    update_initial_vars
    compute_delta_height
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
    epsilon = 10  # max pixel distance to count as a vertex hit
    margin = 10

    def __init__(self, ax, poly, markevery=None, annotate=True):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly
        self.annotate = annotate
        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True, markevery=markevery)
        if self.annotate:
            textc = 'black'
            self.TL = self.ax.text(x[0] + self.margin, y[0] + self.margin, 'TL', animated=True, color=textc)
            self.TR = self.ax.text(x[1] + self.margin, y[1] + self.margin, 'TR', animated=True, color=textc)
            self.BR = self.ax.text(x[2] + self.margin, y[2] + self.margin, 'BR', animated=True, color=textc)
            self.BL = self.ax.text(x[3] + self.margin, y[3] + self.margin, 'BL', animated=True, color=textc)
        elif not self.annotate:
            textc = 'black'
            self.high = self.ax.text(x[0] + self.margin, y[0] + self.margin, 'high', animated=True, color=textc)
            self.low = self.ax.text(x[1] + self.margin, y[1] + self.margin, 'low', animated=True, color=textc)
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
        if self.annotate:
            self.ax.draw_artist(self.TL)
            self.ax.draw_artist(self.TR)
            self.ax.draw_artist(self.BR)
            self.ax.draw_artist(self.BL)
        elif not self.annotate:
            self.ax.draw_artist(self.low)
            self.ax.draw_artist(self.high)
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
        # are we in the current axes
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata
        ind = self._ind

        self.poly.xy[self._ind] = x, y
        self.line.set_data(zip(*self.poly.xy))
        self.canvas.restore_region(self.background)
        if self.annotate:
            logger.info("ind is currently %s -> %s", self._ind, ind)
            if self._ind == 0:
                self.TL.set_position((x + self.margin, y + self.margin))
            if self._ind == 1:
                self.TR.set_position((x + self.margin, y + self.margin))
            if self._ind == 2:
                self.BR.set_position((x + self.margin, y + self.margin))
            if self._ind == 3:
                self.BL.set_position((x + self.margin, y + self.margin))
            self.ax.draw_artist(self.TL)
            self.ax.draw_artist(self.TR)
            self.ax.draw_artist(self.BL)
            self.ax.draw_artist(self.BR)
        elif not self.annotate:
            if self._ind == 0:
                self.high.set_position((x + self.margin, y + self.margin))
            if self._ind == 1:
                self.low.set_position((x + self.margin, y + self.margin))
            self.ax.draw_artist(self.low)
            self.ax.draw_artist(self.high)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        # update our little axis
        self.canvas.blit(self.ax.bbox)



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
        update_initial_vars(self.data, self.model)
        if self.path.exists():
            with open(str(self.path)) as f:
                self.old_calibration = json.load(f)
        else:
            self.old_calibration = {}

    def make_window(self):
        self.fig, self.axes = plt.subplots(2, 3)
        # sic show instructions in the title
        self.fig.suptitle('select 4 points (clockwise start at top left)')
        self.axes[0, 0].set_title("1) projector box")
        self.axes[0, 1].set_title("2) model domain")
        self.axes[1, 0].set_title("3) set low and high")
        self.axes[1, 1].set_title("4) Initial result within the box")
        self.axes[1, 2].set_title("5) initial view of sandbox of entire domain")

    def save(self):
        with open(str(self.path), 'w') as f:
            json.dump(self.result, f, indent=2, cls=NumpyEncoder)

    @property
    def z(self):
        z = self.data['HEIGHT_NODES']
        return z.min(), z.max()

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

    def show_result(self, ax, cbar=True):
        try:self.cb1.remove()
        except Exception: pass
        ax.clear()
        # we should have results by now
        # save the calibration info
        data = self.data
        result = self.result

        # add the transforms
        result.update(compute_transforms(result))

        xy_node = np.c_[
            data['X_NODES'].ravel(),
            data['Y_NODES'].ravel(),
            np.ones_like(data['X_NODES'].ravel())
        ].astype('float32')

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

        heights = calibrated_height_images(
            self.z_values,
            self.z,
            anomaly_name=pathlib.Path('anomaly.npy').absolute()
        )
        height = next(heights)

        data = self.data
        self.data['height'] = height
        warped_height = cv2.warpPerspective(
            height,
            np.array(result['img2box']),
            height.shape[::-1]
        )

        #self.data['height'] = self.sandbox_height
        plot = ax.imshow(warped_height,
            cmap='jet',
            vmin=self.z[0],
            vmax=self.z[-1],

        )
        if cbar:
            self.cb1 = plt.colorbar(plot, ax=ax)
        plt.show()
        return result

    def show_data(self, ax, result):
        try:self.cb2.remove()
        except Exception: pass
        ax.clear()
        img_bbox = matplotlib.path.Path([
            (40, 40),
            (40, 440),
            (600, 440),
            (600, 40)
        ])
        data = self.data
        model_bbox = matplotlib.path.Path(self.model_points)
        data['node_in_box'] = model_bbox.contains_points(np.c_[data['xk'], data['yk']])
        xk_box, yk_box = transform(data['xk'], data['yk'], result['model2box'])
        data['node_in_img_bbox'] = img_bbox.contains_points(np.c_[xk_box, yk_box])
        idx = np.logical_and(data['node_in_box'], data['node_in_img_bbox'])

        data['model2img'] = cv2.getPerspectiveTransform(
            np.array(self.model_points, dtype='float32'),
            np.array(self.img_points, dtype='float32')
        )

        zk_copy = data['zk'].copy()
        self.delta_zk = compute_delta_zk(data, idx)
        zk_copy[idx] += self.delta_zk
        self.axes[1, 2].set_title((str(min(zk_copy)) + " and " + str(max(zk_copy))))

        plot = ax.scatter(
            data['xk'].ravel(),
            data['yk'].ravel(),
            c=zk_copy.ravel(),
            cmap='jet',
            edgecolor='none',
            vmin=self.z[0],
            vmax=self.z[-1]
        )
        self.cb2 = plt.colorbar(plot, ax=ax)
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
        annotate = True
        if len(xs) == 2:
            markevery = [1]
            annotate = False
        p = PolygonInteractor(ax, poly, markevery=markevery, annotate=annotate)
        return p

    def min_slider(self, val):
        self.z_values[0] = val
        self.save()
        result = self.show_result(self.axes[1, 1])
        self.show_data(self.axes[1, 2], result)
        self.set_text()
        self.show_result(self.fig2ax, cbar=False)


    def max_slider(self, val):
        self.z_values[1] = val
        self.save()
        result = self.show_result(self.axes[1, 1])
        self.show_data(self.axes[1, 2], result)
        self.set_text()
        self.show_result(self.fig2ax, cbar=False)

    def set_text(self):
        try: self.text_min.remove()
        except Exception: pass
        try: self.text_max.remove()
        except Exception: pass
        self.text_min = self.fig.text(0.7, 0.7, 'Largest distance (low): ' + str(self.z_values[0]) +
                                        '\n Converted to depth (min Z): ' + str(round(self.delta_zk[0], 2)))
        self.text_max = self.fig.text(0.7, 0.6, 'Smallest distance (high): ' + str(self.z_values[1]) +
                                        '\n Converted to depth (max Z): ' + str(round(self.delta_zk[1], 2)))

    def run(self):
        fig, axes = self.fig, self.axes
        mng = plt.get_current_fig_manager()
        # try and maximize
        try:
            mng.window.showMaximized()
        except AttributeError:
            try:
                mng.resize(*mng.window.maxsize())
            except AttributeError:
                logging.warn('could not maximize, unknown interface')
        # get video and depth image
        video = next(self.videos)
        raw = next(self.raws)

        # show the depth and video in the left window

        plot1 = axes[0, 0].imshow(raw) #, cmap='jet', vmin=650, vmax=800)
        plot2 = axes[1, 0].imshow(raw)
        plt.colorbar(plot1, ax=self.axes[0, 0])
        plt.colorbar(plot2, ax=self.axes[1, 0])

        # convert to array we can feed into opencv
        data = self.data
        scat1 = axes[0, 1].scatter(
            data['X_NODES'].ravel(),
            data['Y_NODES'].ravel(),
            c=data['HEIGHT_NODES'].ravel(),
            cmap='viridis',
            edgecolor='none'
        )
        plt.colorbar(scat1, ax=axes[0, 1])

        img_points = self.old_calibration.get("img_points", 4)
        img_poly = self.add_edit_polygon(axes[0, 0], points=img_points)
        model_points = self.old_calibration.get("model_points", 4)
        model_poly = self.add_edit_polygon(axes[0, 1], points=model_points)
        height_points = self.old_calibration.get("height_points", 2)
        height_poly = self.add_edit_polygon(axes[1, 0], points=2)

        z_values = self.z_values

        msg = """1) shows the raw kinect image, use the dots to select the area
(TL = Top Left, TR = Top Right, BL = Bottom Left, BR = Bottom Right)
2) Shows the domain to select within the bathymetry of the model
3) Select the lowest (red dot) and highest point in the raw kinect data
The lowest point should be with the highest value (distance from kinect to bottom)
4) This box shows the result of the cut out with the polygon
5) Shows the initial calibrated bathymetry of the entire model.
"""
        self.axes[0, 2].text(0, 0.7, msg)
        self.axes[0, 2].axis('off')

        # keep track of the selected points\n
        height_points = self.height_points

        # pointer event
        pid = None

        #Generate figures and sliders at first enter pressed
        self.firstenter = True
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

                if not self.z_values[0] > self.z_values[1]:
                    axes[1, 2].set_title('Choose a min z lower than high z')
                    logger.warn("I had to reverse your z points, please select lower value with the bottom left dot. Coordinates: %s, raw: %s.",
                                self.height_points,
                                self.z_values )
                    self.z_values = list(reversed(self.z_values))
                    self.height_points = list(reversed(self.height_points))
                self.save()
                result = self.show_result(self.axes[1, 1])
                self.show_data(self.axes[1, 2], result)
                if self.firstenter:
                    minz = self.z_values[0]
                    maxz = self.z_values[1]
                    self.slidermin = Slider(plt.axes([0.70, 0.65, 0.15, 0.03]), 'min z',
                                            minz - 50, minz + 50, valinit = minz)
                    self.slidermax = Slider(plt.axes([0.70, 0.55, 0.15, 0.03]), 'max z',
                                            maxz - 50, maxz + 50, valinit = maxz)
                    self.set_text()
                    self.slidermin.on_changed(self.min_slider)
                    self.slidermax.on_changed(self.max_slider)
                    self.secondfig, self.fig2ax = plt.subplots()
                    self.secondfig.subplots_adjust(
                        left=0,
                        right=1,
                        bottom=0,
                        top=1
                    )
                    self.fig2ax.axis('off')
                    self.firstenter = False
                self.show_result(self.fig2ax, cbar=False)


            if event.key =='escape':
                fig.canvas.mpl_disconnect(pid)
                plt.close(fig)
                plt.close(self.secondfig)

        plt.ion()
        pid = fig.canvas.mpl_connect('key_press_event', picker)
        plt.show(block=True)
