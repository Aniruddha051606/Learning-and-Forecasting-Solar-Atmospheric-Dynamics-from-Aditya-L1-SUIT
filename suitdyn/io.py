import numpy as np
from astropy.io import fits


def read(path):
    """Image as float32 (BZERO/BSCALE applied) and its header."""
    with fits.open(path, memmap=False) as h:
        return h[0].data.astype(np.float32), h[0].header


def scale_of(shape):
    """Linear size relative to a 2048 binned frame: 1 for 2048², 2 for 4096², used to scale pixel settings."""
    return shape[0] / 2048
