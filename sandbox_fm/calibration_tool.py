import pathlib
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .sandbox_fm import (
    update_delft3d_initial_vars
)


class RectBuilder:
    def __init__(self, rect):
        self.rect = rect
        self.xs = list(rect.get_xdata())
        self.ys = list(rect.get_ydata())
        self.cid = rect.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self.rect.axes:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.rect.set_data(self.xs, self.ys)
        self.rect.figure.canvas.draw()


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
            json.dump(self.result, f, indent=2)

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

    def save_and_show_result(self, ax):
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

    def run(self):
        fig, axes = self.fig, self.axes

        # get video and depth image
        video = next(self.videos)
        raw = next(self.raws)

        # show the depth and video in the left window
        axes[0, 0].imshow(raw, cmap='viridis')
        axes[0, 0].imshow(video, cmap='Greys', alpha=0.5)
        axes[1, 0].imshow(raw)

        # keep track of the selected points
        img_points = self.img_points
        model_points = self.model_points
        height_points = self.height_points
        z_values = self.z_values


        # pointer event
        pid = None

        # define the point selector
        def picker(event):
            if (event.inaxes == axes[0, 0] and len(img_points) < 4):
                img_points.append((event.xdata, event.ydata))
                event.inaxes.set_title('%s points selected' % (
                    len(img_points), )
                )
            elif (event.inaxes == axes[0, 1] and len(model_points) < 4):
                model_points.append((event.xdata, event.ydata))
                event.inaxes.set_title('%s points selected' % (
                    len(model_points), )
                )
            elif (event.inaxes == axes[1, 0] and len(height_points) < 2):
                height_points.append((event.xdata, event.ydata))
                z_values.append(float(raw[int(event.ydata), int(event.xdata)]))
                title = "%s points selected" % (len(height_points), )
                if (len(height_points) == 0):
                    title = "select a point at -8m"
                elif (len(height_points) == 1):
                    title = "select a point at 12m"
                event.inaxes.set_title(title)
                event.inaxes.plot(event.xdata, event.ydata, 'ko')
                text = "d: %.2f\n(%s, %s)" % (
                    raw[int(event.ydata), int(event.xdata)],
                    int(event.xdata),
                    int(event.ydata)
                )
                event.inaxes.text(
                    event.xdata + 0.5,
                    event.ydata + 0.5,
                    text
                )
            if (
                    len(img_points) == 4
                    and len(model_points) == 4
                    and len(height_points) == 2
            ):
                # stop listening we're done
                fig.canvas.mpl_disconnect(pid)
                self.save_and_show_result(axes[1, 1])


        # convert to array we can feed into opencv
        data = self.data
        axes[0, 1].scatter(
            data['xk'].ravel(),
            data['yk'].ravel(),
            c=data['zk'].ravel(),
            cmap='Greens',
            edgecolor='none'
        )
        plt.ion()
        pid = fig.canvas.mpl_connect('button_press_event', picker)
        plt.show(block=True)
