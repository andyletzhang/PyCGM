import numpy as np

from .calibration import retrieve_first_order
from .fft import FFT_Processor


class CGM_Processor:
    def __init__(self, ref=None, shape=(2160, 2560), gamma=39e-6, d=5e-4, p=6.5e-6, Z=1, threads=4):
        """
        Initialize the CGM_Processor.

        Parameters:
        -----------
        ref : numpy.ndarray, optional
            Reference image to initialize the processor.
        shape : tuple of int, optional
            Shape of the images (if ref is not provided). Defaults to (2160, 2560) (the size of the Phasics SID4 camera chip).
        gamma : float, optional
            The gamma parameter for processing, by default 39e-6
        d : float, optional
            The distance 'd' parameter, by default 5e-4
        p : float, optional
            The pixel size 'p' parameter, by default 6.5e-6
        Z : int, optional
            The Z factor for processing, by default 1
        threads : int, optional
            The number of threads for FFT processing, by default 4

        Notes:
        ------
        If `ref` is provided, the shape is inferred from the reference image.
        """
        if ref is not None:
            shape = ref.shape

        self.Ny, self.Nx = shape
        self.xx, self.yy = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny))
        self.gamma = gamma
        self.d = d
        self.p = p
        self.Z = Z
        self.alpha = gamma / (4 * np.pi * d)
        self.threads = threads

        self.fft_processor = FFT_Processor(shape, threads=threads)

        if ref is not None:
            self.set_reference(ref)

    def set_reference(self, ref):
        """
        Set the reference image for the OPD processor.

        Parameters:
        -----------
        ref : numpy.ndarray
            The reference image to use for processing
        """
        self.ref = ref
        self.FRef = self.fft_processor.fft(ref)
        self.cropsX = retrieve_first_order(np.abs(self.FRef))
        self.cropsY = self.cropsX.rotate90()

        HRef = []
        for c in [self.cropsX, self.cropsY]:
            HRef.append(np.roll(self.FRef * c.circ_mask, shift=(-c.shifty, -c.shiftx), axis=(0, 1)))
        self.IRefx = self.fft_processor.ifft(HRef[0])
        self.IRefy = self.fft_processor.ifft(HRef[1])

    def process(self, itf, ref=None):
        """
        Calculate the OPD map from an interference and reference image.

        Parameters:
        -----------
        itf : numpy.ndarray
            Interference image
        ref : numpy.ndarray
            Reference image

        Returns:
        --------
        numpy.ndarray
            The calculated OPD map
        """
        if ref is not None and not np.array_equal(ref, self.ref):
            self.set_reference(ref)

        # Compute FFTs
        FItf = self.fft_processor.fft(itf)

        # Apply elliptical mask and shift
        H = []
        for c in [self.cropsX, self.cropsY]:
            H.append(np.roll(FItf * c.circ_mask, shift=(-c.shifty, -c.shiftx), axis=(0, 1)))

        # Inverse FFT to get filtered images
        Ix = self.fft_processor.ifft(H[0])
        Iy = self.fft_processor.ifft(H[1])

        # Calculate phase differences
        dw1 = np.angle(np.conjugate(self.IRefx) * Ix) * self.alpha
        dw2 = np.angle(np.conjugate(self.IRefy) * Iy) * self.alpha

        # Rotate phase differences to original coordinate system
        dwX = self.cropsX.angle['cos'] * dw1 - self.cropsX.angle['sin'] * dw2
        dwY = self.cropsX.angle['sin'] * dw1 + self.cropsX.angle['cos'] * dw2

        # Calculate frequency domain coordinates
        kx = self.xx - self.Nx / 2
        ky = self.yy - self.Ny / 2

        # Compute denominator for integration, handling division by zero
        denom = 1j * 2 * np.pi * (kx / self.Nx + 1j * ky / self.Ny)
        denom[np.abs(denom) < 1e-10] = np.inf

        # Integrate phase gradients
        W0 = self.fft_processor.ifft((self.fft_processor.fft(dwX) + 1j * self.fft_processor.fft(dwY)) / denom)

        # Return real part scaled by pixel size and Z factor
        return np.real(W0) * self.p / self.Z

def phase_cmap():
    """
    Load the phase1024.txt colormap and create a LinearSegmentedColormap.
    
    Returns:
    --------
    cmap : LinearSegmentedColormap
        The colormap created from the phase1024.txt file.
    """
    import pkg_resources
    from matplotlib.colors import LinearSegmentedColormap
    
    # Get the path to the phase1024.txt file, bundled in the package
    colormap_file = pkg_resources.resource_filename('pycgm', 'phase1024.txt')
    
    # Load the colormap data
    phase_LUT = np.loadtxt(colormap_file)
    
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list('phase', phase_LUT)
    
    return cmap