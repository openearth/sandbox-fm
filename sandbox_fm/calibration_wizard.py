import pathlib
import json
import logging

import cv2
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt

from .calibrate import (
    compute_transforms
)
from .sandbox_fm import (
    update_initial_vars,
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
        self.count = 1
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
        if hasattr(self, 'cb1'):
            try:
                self.cb1.remove()
            except KeyError:
                # not sure why this happens
                pass
        ax.clear()
        # we should have results by now
        # save the calibration info
        data = self.data

        # add the transforms
        data.update(compute_transforms(self.result))

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
                np.array(data['model2box'], dtype='float32')
            )
        )

        kinect_heights = calibrated_height_images(
            self.z_values,
            self.z,
            anomaly_name=pathlib.Path('anomaly.npy').absolute()
        )
        kinect_height = next(kinect_heights)
        self.data['kinect_height'] = kinect_height

        warped_height = cv2.warpPerspective(
            kinect_height,
            np.array(data['img2box']),
            kinect_height.shape[::-1]
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

    def show_data(self, ax):
        if hasattr(self, 'cb2'):
            try:
                self.cb2.remove()
            except KeyError:
                # not sure why we get this error
                pass


        ax.clear()
        img_bbox = matplotlib.path.Path([
            (40, 40),
            (40, 440),
            (600, 440),
            (600, 40)
        ])
        data = self.data
        data.update(compute_transforms(self.result))

        model_bbox = matplotlib.path.Path(self.model_points)
        data['node_in_box'] = model_bbox.contains_points(
            np.c_[
                data['X_NODES'].ravel(),
                data['Y_NODES'].ravel()
            ]
        )
        x_box, y_box = transform(
            data['X_NODES'].ravel(),
            data['Y_NODES'].ravel(),
            data['model2box']
        )
        data['node_in_img_bbox'] = img_bbox.contains_points(
            np.c_[
                x_box,
                y_box
            ]
        )
        idx = np.logical_and(data['node_in_box'], data['node_in_img_bbox'])
        # reshape to original shape
        idx = idx.reshape(data['X_NODES'].shape)

        data['model2img'] = cv2.getPerspectiveTransform(
            np.array(self.model_points, dtype='float32'),
            np.array(self.img_points, dtype='float32')
        )

        height_copy = data['HEIGHT_NODES'].copy()
        self.delta_height = compute_delta_height(data, idx)
        height_copy[idx] += self.delta_height
        title = "{} and {}".format(height_copy.min(), height_copy.max())
        plot = ax.scatter(
            data['X_NODES'].ravel(),
            data['Y_NODES'].ravel(),
            c=height_copy.ravel(),
            cmap='jet',
            edgecolor='none',
            vmin=self.z[0],
            vmax=self.z[-1]
        )
        # self.cb2 = plt.colorbar(plot, ax=ax)
        plt.show()

    def run(self):
        mng = plt.get_current_fig_manager()
        # try and maximize
        try:
            mng.window.showMaximized()
        except AttributeError:
            try:
                mng.resize(*mng.window.maxsize())
            except AttributeError:
                logging.warn('could not maximize, unknown interface')

        self.raws = next(self.raws)
        self.videos = next(self.videos)
        # initiliaze the first window, where self.count = 1
        self.update_window()
        plt.ion()
        plt.show(block=True)

    def make_window(self):
        self.fig = plt.figure()
        self.titleAx = self.fig.add_axes([0.1, 0.8, 0.8, 0.15])
        self.titleAx.axis('off')

        self.plotAxLeft = self.fig.add_axes([0.1, 0.2, 0.4, 0.6])
        self.plotAxRight = self.fig.add_axes([0.5, 0.2, 0.4, 0.6])
        self.prevAx = self.fig.add_axes([0.3, 0.05, 0.1, 0.05])
        self.nextAx = self.fig.add_axes([0.6, 0.05, 0.1, 0.05])
        self.saveExitAx = self.fig.add_axes([0.8, 0.05, 0.1, 0.05])
        self.sliderminAx = self.fig.add_axes([0.60, 0.5, 0.2, 0.03])
        self.slidermaxAx = self.fig.add_axes([0.60, 0.3, 0.2, 0.03])

        self.bnext = Button(self.nextAx, 'Next')
        self.bprev = Button(self.prevAx, 'Previous')
        self.bsave = Button(self.saveExitAx, 'Save and Exit')

        def callback_next(event):
            self.count += 1
            self.update_window()
        self.bnext.on_clicked(callback_next)

        def callback_previous(event):
            self.count -= 1
            self.update_window()
        self.bprev.on_clicked(callback_previous)

        def callback_save(event):
            self.save()
            plt.close(self.fig)
            plt.close(self.secondfig)
        self.bsave.on_clicked(callback_save)

    def update_window(self):
        # Clean the figure up a bit
        # When initializing update_window, colorbars are note yet created
        self.titleAx.clear()
        self.plotAxLeft.clear()
        self.plotAxRight.clear()
        self.fig.delaxes(self.plotAxLeft)
        self.fig.delaxes(self.plotAxRight)

        plt.draw()
        try:
            plt.close(self.secondfig)
        except Exception:
            pass

        self.plotAxLeft = self.fig.add_axes([0.1, 0.2, 0.4, 0.6])
        self.plotAxRight = self.fig.add_axes([0.5, 0.2, 0.4, 0.6])

        if (self.count == 1):
            self.prevAx.set_visible(False)
            self.nextAx.set_visible(True)
            self.saveExitAx.set_visible(False)
            self.sliderminAx.set_visible(False)
            self.slidermaxAx.set_visible(False)
            self.bprev.active = False
            self.bnext.active = True
            self.bsave.active = False
            self.titleAx.axis('off')
            self.plotAxRight.axis('off')
            self.plotAxLeft.axis('off')

            self.titleAx.text(0.4, 0.5, """
                                Step 1: The image to the left shows the raw image of the kinect device. Drag the corners of the
                                polygon to cut out the section to be used from the raw image. The corners should all lie between
                                the borders of the box. The image to the right is an photo made by the kinect and can be used
                                as a guide to identify what is seen by the raw data.
                                """,
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=20)
            self.plot1 = self.plotAxLeft.imshow(self.raws)
            # self.cbLeft = plt.colorbar(self.plot1, ax=self.plotAxLeft)

            self.plot2 = self.plotAxRight.imshow(self.videos)

            img_points = self.old_calibration.get("img_points", 4)
            self.img_poly = self.add_edit_polygon(self.plotAxLeft, points=img_points)
            plt.draw()

        elif (self.count == 2):
            self.prevAx.set_visible(True)
            self.nextAx.set_visible(True)
            self.saveExitAx.set_visible(False)
            self.sliderminAx.set_visible(False)
            self.slidermaxAx.set_visible(False)
            self.bprev.active = True
            self.bnext.active = True
            self.bsave.active = False
            self.titleAx.axis('off')
            self.plotAxRight.axis('off')
            self.plotAxLeft.axis('off')

            self.titleAx.text(0.4, 0.5, """
                                Step 2: Now select which part of the model should be used. To the left you see the image cut out in
                                step 1. to the right you see the entire model domain. Use to corners of the polygon to cut out the
                                section to be used.
                                """,
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=20)

            self.plotLeft = self.plotAxLeft.imshow(self.raws)
            # self.cbLeft = plt.colorbar(self.plotLeft, ax=self.plotAxLeft)
            self.plotAxLeft.text(1,1, 'TR',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=self.plotAxLeft.transAxes)
            self.plotAxLeft.text(0, 1, 'TL',
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=self.plotAxLeft.transAxes)
            self.plotAxLeft.text(1, 0, 'BR',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=self.plotAxLeft.transAxes)
            self.plotAxLeft.text(0, 0, 'BL',
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=self.plotAxLeft.transAxes)
            self.plotRight = self.plotAxRight.scatter(
                self.data['X_NODES'].ravel(),
                self.data['Y_NODES'].ravel(),
                c=self.data['HEIGHT_NODES'].ravel(),
                cmap='viridis',
                edgecolor='none'
            )
            # self.cbRight = plt.colorbar(self.plotRight, ax=self.plotAxRight)
            model_points = self.old_calibration.get("model_points", 4)
            print('model points', model_points)
            self.model_poly = self.add_edit_polygon(self.plotAxRight, points=model_points)
            plt.draw()

        elif (self.count == 3):
            self.save()
            self.plotAxRight.clear()
            self.plotAxRight.set_visible(False)
            self.prevAx.set_visible(True)
            self.nextAx.set_visible(False)
            self.saveExitAx.set_visible(True)
            self.sliderminAx.set_visible(True)
            self.slidermaxAx.set_visible(True)
            self.bprev.active = True
            self.bnext.active = False
            self.bsave.active = True
            self.titleAx.axis('off')
            self.plotAxLeft.axis('off')

            self.titleAx.text(0.4, 0.5, """
                                Step 3: If necessary the height of the cut out section can be adjusted to correctly match the height
                                in the model. Use the sliders to change the min and max height of the cut out section.
                                """,
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=20)


            self.img_points = list(zip(
                *self.img_poly.line.get_data()
            ))
            self.model_points = list(zip(
                *self.model_poly.line.get_data()
            ))

            self.z_values = [self.raws.max(), self.raws.min()]
            self.rangeminz = self.z_values[0]
            self.rangemaxz = self.z_values[1]

            self.slidermin = Slider(self.sliderminAx, 'min',
                                    self.rangeminz - 50, self.rangeminz + 50, valinit = self.z_values[0])
            self.slidermax = Slider(self.slidermaxAx, 'max',
                                    self.rangemaxz - 50, self.rangemaxz + 50, valinit = self.z_values[1])
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
            self.show_data(self.plotAxLeft)
            plt.draw()

    def min_slider(self, val):
        self.z_values[0] = val
        self.save()
        self.show_result(self.fig2ax, cbar=False)
        self.show_data(self.plotAxLeft)

    def max_slider(self, val):
        self.z_values[1] = val
        self.save()
        self.show_result(self.fig2ax, cbar=False)
        self.show_data(self.plotAxLeft)

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
            facecolor='blue',
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

if __name__=='__main__':
    """calibrate the sandbox by selecting both 4 points in box and in model"""

    schematization_path = pathlib.Path(schematization.name)
    path = schematization_path.with_name('calibration.json').absolute()
    # raw images
    videos = video_images()
    raws = depth_images()
    # start the model (changes directory)
    model = bmi.wrapper.BMIWrapper(engine)
    # this stores current path

    # this changes directory
    model.initialize(str(schematization_path.absolute()))

    calibration = Calibration(path, videos, raws, model)
    calibration.run()
