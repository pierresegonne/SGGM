import math
import time
import numpy as np

from torch import cat, matmul


def batchify(*arrays, batch_size=10, shuffel=True):
    """Function that defines a generator that keeps outputting new batches
        for the input arrays infinit times.
    Arguments:
        *arrays: a number of arrays all assume to have same length along the
            first dimension
        batch_size: int, size of each batch
        shuffel: bool, if the arrays should be shuffeled when we have reached
            the end
    """
    N = arrays[0].shape[0]
    c = -1
    while True:
        c += 1
        if c * batch_size >= N:  # reset if we reach end of array
            c = 0
            if shuffel:
                perm_idx = np.random.permutation(N)
                arrays = [a[perm_idx] for a in arrays]
        lower = c * batch_size
        upper = (c + 1) * batch_size
        yield [a[lower:upper] for a in arrays]


def dist(X, Y):  # X:  N x d , Y: M x d
    dist = (
        X.norm(p=2, dim=1, keepdim=True) ** 2
        + Y.norm(p=2, dim=1, keepdim=False) ** 2
        - 2 * matmul(X, Y.t())
    )
    return dist.clamp(0.0)  # N x M


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


def normal_log_prob(x, mean, var):
    print(x.shape, mean.shape, var.shape)
    c = -0.5 * math.log(2 * math.pi)
    assert np.array_equal(x.shape, mean.shape)
    assert np.array_equal(x.shape, var.shape)
    if isinstance(x, np.ndarray):  # numpy implementation
        return (
            c
            - np.log(var) / 2
            - (x - mean) ** 2 / (2 * var)
        )
    else:  # torch implementation
        return (
            c
            - var.log() / 2
            - (x - mean) ** 2 / (2 * var)
        )


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
