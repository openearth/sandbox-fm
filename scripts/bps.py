#!/usr/bin/env python
import functools

import numpy as np
import mmi
import mmi.mmi_client

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.animation
import threading

WIDTH = 1000
HEIGHT = 1000

def create_fig():
    """create a figure with axes"""
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.autoscale('off')
    img = plt.imread('https://www.visserenvisser.nl/media/catalog/product/cache/1/image/c337cbb1759470c48c1849f8095cafa7/h/3/h360-peilschaal-emaille-zonder-logo.png')
    ax.imshow(img)
    return fig, ax


def init(ln):
    """create a line"""
    ln.set_data([], [])
    return ln,


def update(frame, ln, ax, data):
    """update the line"""
    # noramlize to imshow coordinates TODO: normalize image
    # range 3 - 6 m
    val = HEIGHT - (matplotlib.colors.Normalize(3, 6, clip=True)(data['s1']) * HEIGHT)
    ln.set_data([0, WIDTH], [val, val])
    ax.set_title("%.2f -> %.0f (%s)" % (data['s1'], val, data['counter']))


def connect_model():
    """connect to a running model, continue and subscribe"""
    model = mmi.mmi_client.MMIClient('tcp://localhost:62000')
    model.remote('play')
    poller = model.subscribe()
    return poller


def update_data(poller, data):
    """update a shared array"""
    while True:
        for sock, n in poller.poll(100):
            for i in range(n):
                message = mmi.recv_array(sock)
                arr, meta = message
                if meta['name'] == 's1':
                    # just pick a value
                    data['counter'] += 1
                    data['s1'] = arr[739]



if __name__ == '__main__':

    data = {
        'counter': 0,
        's1': 0
    }
    # connect to moddel
    poller = connect_model()

    # start the data update in the background
    threading.Thread(
        target=functools.partial(
            update_data,
            data=data,
            poller=poller
        )
    ).start()
    fig, ax = create_fig()

    # we need this part global
    ln, = ax.plot(
        [0, WIDTH],
        [HEIGHT/2, HEIGHT/2],
        'k-'
    )

    animation = matplotlib.animation.FuncAnimation(
        fig,
        functools.partial(update, data=data, ln=ln, ax=ax),
        frames=np.linspace(0, 2*np.pi, 128),
        init_func=functools.partial(init, ln=ln)
    )
    plt.show()
