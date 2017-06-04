import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
import matplotlib as mpl
import numpy as np
from scipy.misc import imsave
from scipy.stats import gaussian_kde, entropy

mpl.rc('figure.subplot',
       left=0.1,
       right=0.9,
       top=0.9,
       bottom=0.1,
       wspace=0.15,
       hspace=0.15)
mpl.rcParams['savefig.dpi'] = 300


class FigureDict():
    def __init__(self):
        self.dict = dict()

    def __getitem__(self, item):
        if item in self.dict.keys():
            return self.dict[item]
        else:
            self.dict[item] = plt.figure()
            self.dict[item].canvas.set_window_title(item)
            return self.dict[item]


def split(x):
    assert type(x) == int
    t = int(np.floor(np.sqrt(x)))
    for a in range(t, 0, -1):
        if x % a == 0:
            return a, x / a


def line_transform(x, size):
    a = x.shape[0]
    h, w = size[0], size[1]
    x = np.reshape(x, [a, h, w])
    x = np.transpose(x, [1, 0, 2])
    x = np.reshape(x, [h, a * w])
    return x


def grid_transform(x, size, dims=None):
    if dims is None:
        a, b = split(x.shape[0])
    else:
        a, b = dims
    h, w = size[0], size[1]
    if len(size) > 2:
        ch = size[2]
    else:
        ch = 0

    if ch:
        x = np.reshape(x, [a, b, h, w, ch])
        x = np.transpose(x, [0, 2, 1, 3, 4])
        x = np.reshape(x, [a * h, b * w, ch])
    else:
        x = np.reshape(x, [a, b, h, w])
        x = np.transpose(x, [0, 2, 1, 3])
        x = np.reshape(x, [a * h, b * w])
    return x


def images_one_step(model, fig, samples, steps, size, data_to_image=None):
    fig.clear()
    #bz = model.noise_sampler(samples)
    step = max(steps)
    #o = model.evaluate(bz, step)
    o = model.evaluate(step)
    while o.shape[0] < samples:
        o = np.concatenate([o, model.evaluate(step)], axis=0)
    o = o[:samples]

    if data_to_image:
        o = data_to_image(o)
    a, b = split(len(steps))
    base = a * 100 + b * 10 + 1
    for i in range(0, len(steps)):
        ax = fig.add_subplot(base + i)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        x = grid_transform(o[:, steps[i]-1], size)
        ax.imshow(x, cmap='gray')
    return o


def images_multiple_steps(model, fig, samples, steps, size):
    fig.clear()
    #bz = model.noise_sampler(samples)
    #o = model.evaluate(bz, steps)
    o = model.evaluate(steps)
    a, b = split(samples)
    base = a * 100 + b * 10
    for i in range(0, samples):
        ax = fig.add_subplot(base + i + 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        x = grid_transform(o[i], size)
        ax.imshow(x, cmap='gray')
    return o


def images_chain(model, fig, steps, size, data_to_image=None):
    fig.clear()
    o = model.evaluate(steps)
    if data_to_image:
        o = data_to_image(o)
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    x = grid_transform(o, size, dims=[o.shape[0], o.shape[1]])
    ax.imshow(x, cmap='gray')
    return x


def images_one_chain(model, fig, steps, size, data_to_image=None):
    fig.clear()
    o = model.evaluate(steps)
    o = o[0:1]
    if data_to_image:
        o = data_to_image(o)
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    x = grid_transform(o, size)
    ax.imshow(x, cmap='gray')
    return x

figs = FigureDict()
