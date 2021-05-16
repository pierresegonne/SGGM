import matplotlib as mpl
import matplotlib.pyplot as plt
import random

from typing import Tuple, Union

# DTU colours
# Source: https://www.designguide.dtu.dk/
colours = {
    "primaryRed": "#990000",
    "white": "#ffffff",
    "black": "#000000",
    "blue": "#2F3EEA",
    "brightGreen": "#1FD082",
    "navyBlue": "#030F4F",
    "yellow": "#F6D04D",
    "orange": "#FC7634",
    "pink": "#F7BBB1",
    "grey": "#DADADA",
    "red": "#E83F48",
    "green": "#008835",
    "purple": "#79238E",
}

colours_rgb = {
    "primaryRed": (153, 0, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "blue": (47, 62, 234),
    "brightGreen": (31, 208, 130),
    "navyBlue": (3, 15, 79),
    "yellow": (246, 208, 77),
    "orange": (252, 118, 52),
    "pink": (247, 187, 177),
    "grey": (218, 218, 218),
    "red": (232, 63, 72),
    "green": (0, 136, 53),
    "purple": (121, 35, 142),
}


def hex_to_rgb(
    hex: str, norm: bool = False
) -> Tuple[Union[int, float], Union[int, float], Union[int, float]]:
    if hex[0] == "#":
        hex = hex[1:]
    div = 255 if norm else 1
    return tuple(int(hex[i : i + 2], 16) / div for i in (0, 2, 4))


def scale_down_rgb(rgb):
    return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


colours_rgb = {k: scale_down_rgb(v) for k, v in colours_rgb.items()}


def random_rgb_colour():
    return random.choice(list(colours_rgb.values()))


# General Style, possibilites
# [
#   'seaborn-ticks','ggplot', 'dark_background', 'bmh', 'seaborn-poster',
#   'seaborn-notebook', 'fast', 'seaborn', 'classic', 'Solarize_Light2',
#   'seaborn-dark', 'seaborn-pastel', 'seaborn-muted', '_classic_test',
#   'seaborn-paper', 'seaborn-colorblind', 'seaborn-bright', 'seaborn-talk',
#   'seaborn-dark-palette', 'tableau-colorblind10', 'seaborn-darkgrid',
#   'seaborn-whitegrid', 'fivethirtyeight', 'grayscale', 'seaborn-white',
#   'seaborn-deep'
# ]
plt.style.use("seaborn-paper")

# Overriden RC Params
rc_params = {
    "font.size": 16,
    # 'text.fontsize': 18,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "figure.figsize": (10, 5),
    "legend.fontsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    # fonts
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
}

for param_key, param_value in rc_params.items():
    mpl.rcParams[param_key] = param_value
