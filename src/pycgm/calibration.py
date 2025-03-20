from dataclasses import dataclass
import numpy as np

try:
    import cupy as cp
    has_gpu = True
except ImportError:
    cp = None
    has_gpu = False

from skimage.feature import peak_local_max

@dataclass
class FcropParameters:
    x: int
    y: int
    R: float
    Nx: int
    Ny: int
    gpu: bool = False
    
    def __post_init__(self):
        # Choose the array library based on GPU flag
        self.xp = cp if self.gpu and has_gpu else np
        
        self.shiftx = round(self.x - (self.Nx / 2 + 1))
        self.shifty = round(self.y - (self.Ny / 2 + 1))
        self._nshiftx = self.shiftx / self.Nx
        self._nshifty = self.shifty / self.Ny
        self._nshiftr = self.xp.hypot(self._nshiftx, self._nshifty)
        self._circ_mask = None
            
        if self._nshiftr == 0:
            self.zeta = float("inf")
            self.angle = {"cos": 1.0, "sin": 0.0}
        else:
            self.zeta = 1 / self._nshiftr
            
            # Handle potential GPU array conversion for angle values
                
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
        return FcropParameters(int(x2), int(y2), self.R, self.Nx, self.Ny, self.gpu)

    @property
    def circ_mask(self):
        if self._circ_mask is None:
            xx, yy = self.xp.meshgrid(self.xp.arange(self.Nx), self.xp.arange(self.Ny))
            self._circ_mask = (
                (xx - self.x) ** 2 / self.Rx**2 + (yy - self.y) ** 2 / self.Ry**2
            ) < 1
        return self._circ_mask


def retrieve_first_order(image, gpu=False):
    """
    Find the first order peak in the FFT image.
    
    Parameters:
    -----------
    image : numpy.ndarray or cupy.ndarray
        FFT magnitude image
    gpu : bool, optional
        Whether to use GPU acceleration if available, by default False
        
    Returns:
    --------
    FcropParameters: 
        Parameters for cropping around the first order peak
    """
    # Convert to numpy if needed for scikit-image processing
    if gpu:
        image_np = image.get()
    else:
        image_np = image
    
    Nx, Ny = image_np.shape[1], image_np.shape[0]
    
    # Find local maxima
    min_distance = min(image_np.shape) // 5
    threshold = 0.1 * image_np.max()
    coordinates = peak_local_max(
        image_np, min_distance=min_distance, threshold_abs=threshold
    )
    
    # Sort by intensity (highest to lowest)
    sorted_coords = sorted(coordinates, key=lambda c: image_np[tuple(c)], reverse=True)
    
    # Zero order is the brightest, first order is second brightest
    zeroth_order, first_order = sorted_coords[:2]
    # Calculate radius
    r = np.linalg.norm(np.array(zeroth_order) - np.array(first_order))
    
    # Return as FcropParameters with GPU flag
    y, x = first_order.astype(int)
    return FcropParameters(x, y, r / 2, Nx, Ny, gpu=gpu)