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
    x: float
    y: float
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
        
        # Convert scalar values for GPU compatibility if needed
        if self.gpu:
            nshiftr_scalar = float(self._nshiftr.get())
        else:
            nshiftr_scalar = float(self._nshiftr)
            
        if nshiftr_scalar == 0:
            self.zeta = float("inf")
            self.angle = {"cos": 1.0, "sin": 0.0}
        else:
            self.zeta = 1 / nshiftr_scalar
            
            # Handle potential GPU array conversion for angle values
            if self.gpu:
                nshiftx_scalar = float(self._nshiftx.get())
                nshifty_scalar = float(self._nshifty.get())
            else:
                nshiftx_scalar = float(self._nshiftx)
                nshifty_scalar = float(self._nshifty)
                
            self.angle = {
                "cos": nshiftx_scalar / nshiftr_scalar,
                "sin": nshifty_scalar / nshiftr_scalar,
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
        return FcropParameters(x2, y2, self.R, self.Nx, self.Ny, self.gpu)

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
    return FcropParameters(*first_order[::-1], r / 2, Nx, Ny, gpu=gpu)