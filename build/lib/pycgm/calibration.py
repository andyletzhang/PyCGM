from dataclasses import dataclass

import numpy as np
from skimage.feature import peak_local_max

@dataclass
class FcropParameters:
    x: float
    y: float
    R: float
    Nx: int
    Ny: int

    def __post_init__(self):
        self.shiftx = round(self.x - (self.Nx / 2 + 1))
        self.shifty = round(self.y - (self.Ny / 2 + 1))

        self._nshiftx = self.shiftx / self.Nx
        self._nshifty = self.shifty / self.Ny
        self._nshiftr = np.hypot(self._nshiftx, self._nshifty)

        self._circ_mask = None

        if self._nshiftr == 0:
            self.zeta = float("inf")
            self.angle = {"cos": 1.0, "sin": 0.0}
        else:
            self.zeta = 1 / self._nshiftr
            self.angle = {
                "cos": self._nshiftx / self._nshiftr,
                "sin": self._nshifty / self._nshiftr,
            }

        self.Rx = self.Nx / self.zeta / 2
        self.Ry = self.Ny / self.zeta / 2

    def rotate90(self):
        Nratio = self.Ny / self.Nx
        x0, y0 = self.Nx / 2 + 1, self.Ny / 2 + 1
        dx, dy = self.x - x0, self.y - y0
        x0 *= Nratio
        dx *= Nratio
        x2 = x0 - dy
        y2 = y0 + dx
        x2 /= Nratio
        return FcropParameters(x2, y2, self.R, self.Nx, self.Ny)

    @property
    def circ_mask(self):
        if self._circ_mask is None:
            xx, yy = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny))
            self._circ_mask = (
                (xx - self.x) ** 2 / self.Rx**2 + (yy - self.y) ** 2 / self.Ry**2
            ) < 1
        return self._circ_mask


def retrieve_first_order(image):
    """
    Find the first order peak in the FFT image.

    Args:
        image (ndarray): FFT magnitude image

    Returns:
        FcropParameters: Parameters for cropping around the first order peak
    """
    Nx, Ny = image.shape[1], image.shape[0]

    # Find local maxima
    min_distance = min(image.shape) // 5
    threshold = 0.1 * image.max()
    coordinates = peak_local_max(
        image, min_distance=min_distance, threshold_abs=threshold
    )

    # Sort by intensity (highest to lowest)
    sorted_coords = sorted(coordinates, key=lambda c: image[tuple(c)], reverse=True)

    # Zero order is the brightest, first order is second brightest
    zeroth_order, first_order = sorted_coords[:2]

    # Calculate radius
    r = np.linalg.norm(np.array(zeroth_order) - np.array(first_order))

    # Return as FcropParameters
    return FcropParameters(*first_order[::-1], r / 2, Nx, Ny)
