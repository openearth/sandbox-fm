#!/usr/bin/env python
import functools

import mmi
import mmi.mmi_client

import matplotlib.pyplot as plt
import threading


if __name__ == '__main__':
    img = plt.imread('https://www.visserenvisser.nl/media/catalog/product/cache/1/image/c337cbb1759470c48c1849f8095cafa7/h/3/h360-peilschaal-emaille-zonder-logo.png')

    # TODO: rewrite into funcanimation
    fig, ax = plt.subplots()
    ax.axis('off')

    model = mmi.mmi_client.MMIClient('tcp://localhost:62000')
    model.remote('play')
    poller = model.subscribe()
    ax.imshow(img)
    ax.autoscale('off')

    # create a line
    ln, = ax.plot(
        [0, 500],
        [10, 10],
        'k-'
    )

    def update_chart():
        while True:
            for sock, n in poller.poll(100):
                for i in range(n):
                    message = mmi.recv_array(sock)
                    arr, meta = message
                    if meta['name'] == 's1':
                        ln.set_data([0, 500], [arr[739]*100, arr[739]*100])
                        ax.plot([500, 1000], [arr[739]*100, arr[739]*100])
                        fig.canvas.draw()
            plt.draw()
    plt.ion()
    threading.Thread(
        target=update_chart
    ).start()
    plt.show(block=True)
    # while True:
