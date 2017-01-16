import pathlib
import json
import logging

import cv2
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment
import matplotlib.pyplot as plt


from .sandbox_fm import (
    update_delft3d_initial_vars
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

    def __init__(self, ax, poly, markevery=None):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True, markevery=markevery)
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
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


class Calibration(object):
    def __init__(self, videos, raws, model):
        self.videos = videos
        self.raws = raws
        self.model = model
        self.data = {}
        self.height_points = []
        self.model_points = []
        self.img_points = []
        self.z_values = []
        # define fixed box coordinate system (what will be on the screen)
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

    def make_window(self):
        self.fig, self.axes = plt.subplots(2, 2)
        # sic show instructions in the title
        self.fig.suptitle('select 4 points (clockwise start at top left)')
        self.axes[0, 0].set_title("projector box")
        self.axes[0, 1].set_title("model box")
        self.axes[1, 0].set_title("low high")
        self.axes[1, 1].set_title("result")

    def save(self):
        with open(str(self.curdir / 'calibration.json'), 'w') as f:
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

        model2box = cv2.getPerspectiveTransform(
            np.array(model_points, dtype='float32'),
            box
        )
        print(img_points, box)
        img2box = cv2.getPerspectiveTransform(
            np.array(img_points, dtype='float32'),
            box
        )
        img2model = cv2.getPerspectiveTransform(
            np.array(img_points, dtype='float32'),
            np.array(model_points, dtype='float32')
        )
        model2img = cv2.getPerspectiveTransform(
            np.array(model_points, dtype='float32'),
            np.array(img_points, dtype='float32')
        )
        box2model = cv2.getPerspectiveTransform(
            np.array(box, dtype='float32'),
            np.array(model_points, dtype='float32')
        )
        box2img = cv2.getPerspectiveTransform(
            np.array(box, dtype='float32'),
            np.array(img_points, dtype='float32')
        )

        comment = """
        This file contains calibrations for model %s.
        It is generated with the perspective transform from opencv.
        """ % (self.model.configfile, )
        result = {
            "model2box": model2box.tolist(),
            "img2box": img2box.tolist(),
            "img2model": img2model.tolist(),
            "model2img": model2img.tolist(),
            "box2model": box2model.tolist(),
            "box2img": box2img.tolist(),
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
            s=20,
            alpha=0.5
        )
        # transformed video on top
        ax.imshow(
            cv2.warpPerspective(
                next(self.videos),
                np.array(result['img2box'], dtype='float32'),
                (640, 480)
            ),
            cmap='Reds',
            alpha=0.5
        )
        ax.set_title('You are done (result below)')
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
        if points == 4:
            # same order as self.box
            xs = [xmin, xmax, xmax, xmin]
            ys = [ymin, ymin, ymax, ymax]
        elif points == 2:
            xs = [xmin, xmax]
            ys = [ymin, ymax]
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
        if points == 2:
            markevery = [1]
        p = PolygonInteractor(ax, poly, markevery=markevery)
        return p

    def run(self):
        fig, axes = self.fig, self.axes

        # get video and depth image
        video = next(self.videos)
        raw = next(self.raws)

        # show the depth and video in the left window
        axes[0, 0].imshow(raw, cmap='viridis')
        axes[0, 0].imshow(video, cmap='Greys', alpha=0.5)
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

        img_poly = self.add_edit_polygon(axes[0, 0])
        model_poly = self.add_edit_polygon(axes[0, 1])
        height_poly = self.add_edit_polygon(axes[1, 0], points=2)

        # keep track of the selected points

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

                fig.canvas.mpl_disconnect(pid)
                self.save()
                self.show_result(axes[1, 1])


        plt.ion()
        pid = fig.canvas.mpl_connect('key_press_event', picker)
        plt.show(block=True)
