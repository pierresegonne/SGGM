import time
import numpy as np

from torch import cat


def ds_from_dl(dl, device):
    for idx, batch in enumerate(iter(dl)):
        _x, _y = batch
        # dm not registered on device
        _x, _y = _x.to(device), _y.to(device)
        if idx == 0:
            x, y = _x, _y
        else:
            x = cat((x, _x), dim=0)
            y = cat((y, _y), dim=0)
    return x, y


class timer(object):
    """ Small class for logging time consumption of models """

    def __init__(self):
        self.timings = []
        self.start = 0
        self.stop = 0

    def begin(self):
        self.start = time.time()

    def end(self):
        self.stop = time.time()
        self.timings.append(self.stop - self.start)

    def res(self):
        print("Total train time: {0:.3f}".format(np.array(self.timings).sum()))
        print("Train time per model: {0:.3f}".format(np.array(self.timings).mean()))
