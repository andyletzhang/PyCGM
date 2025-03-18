import numpy as np

from .calibration import retrieve_first_order

try:
    import cupy as cp
    has_gpu = True
except ImportError:
    cp = None
    has_gpu = False

class CGM_Processor:
    def __init__(self, ref=None, shape=(2160, 2560), gamma=39e-6, d=5e-4, p=6.5e-6, Z=1, gpu=True):
        """
        Initialize the CGM_Processor.

        Parameters:
        -----------
        ref : numpy.ndarray or cupy.ndarray, optional
            Reference image to initialize the processor.
        shape : tuple of int, optional
            Shape of the images (if ref is not provided). Defaults to (2160, 2560) (the size of the Phasics SID4 camera chip).
        gamma : float, optional
            Period of the cross grating in meters, by default 39e-6
        d : float, optional
            Grating-camera distance in meters, by default 5e-4
        p : float, optional
            Camera pixel size in meters, by default 6.5e-6
        Z : int, optional
            Zoom of the relay lens, by default 1
            
        gpu : bool, optional
            Whether to use GPU acceleration if available, by default True

        Notes:
        ------
        If `ref` is provided, the shape is inferred from the reference image.
        """
        if ref is not None:
            shape = ref.shape
        
        self.gpu = has_gpu and gpu
        # Choose appropriate array library based on GPU availability
        self.xp = cp if self.gpu else np
        
        self.Ny, self.Nx = shape
        
        # Create mesh grid on appropriate device
        self.xx, self.yy = self.xp.meshgrid(self.xp.arange(self.Nx), self.xp.arange(self.Ny))
        
        self.gamma = gamma
        self.d = d
        self.p = p
        self.Z = Z
        self.alpha = gamma / (4 * self.xp.pi * d)

        if ref is not None:
            self.set_reference(ref)

    def fft(self, image):
        """ Compute shifted FFT of an image. """
        return self.xp.fft.fftshift(self.xp.fft.fft2(image))

    def ifft(self, image):
        """ Compute inverse FFT of an image. """
        return self.xp.fft.ifft2(self.xp.fft.ifftshift(image))

    def set_reference(self, ref):
        """ Set the reference image for the OPD processor. """
        # Ensure reference is on the correct device
        self.ref = self._asarray(ref)
            
        self.FRef = self.fft(self.ref)
        
        # Get magnitude and pass GPU flag to retrieve_first_order
        FRef_abs = self.xp.abs(self.FRef)
            
        # Pass GPU flag to retrieve_first_order
        self.cropsX = retrieve_first_order(FRef_abs, gpu=self.gpu)
        self.cropsY = self.cropsX.rotate90()  # rotate90 now passes GPU flag

        HRef = []
        for c in [self.cropsX, self.cropsY]:
            # circ_mask is now created with the right array library
            HRef.append(self.xp.roll(self.FRef * c.circ_mask, shift=(-c.shifty, -c.shiftx), axis=(0, 1)))
            
        self.IRefx = self.ifft(HRef[0])
        self.IRefy = self.ifft(HRef[1])

    def _asarray(self, arr):
        """ Ensure the array is on the correct device (GPU or CPU). """
        if self.gpu:
            return cp.asarray(arr)
        else:
            return arr

    def process(self, itf, ref=None):
        """
        Calculate the OPD map from an interference and reference image.

        Parameters:
        -----------
        itf : numpy.ndarray or cupy.ndarray
            Interference image
        ref : numpy.ndarray or cupy.ndarray, optional
            Reference image

        Returns:
        --------
        numpy.ndarray or cupy.ndarray
            The calculated OPD map
        """
        # Ensure input is on correct device
        itf = self._asarray(itf)
        
        if ref is not None:
            ref = self._asarray(ref)
            if not self.xp.array_equal(ref, self.ref):
                self.set_reference(ref)

        # Compute FFTs
        FItf = self.fft(itf)

        # Apply elliptical mask and shift
        H = []
        for c in [self.cropsX, self.cropsY]:
            # circ_mask is already on the right device
            H.append(self.xp.roll(FItf * c.circ_mask, shift=(-c.shifty, -c.shiftx), axis=(0, 1)))

        # Inverse FFT to get filtered images
        Ix = self.ifft(H[0])
        Iy = self.ifft(H[1])

        # Calculate phase differences
        dw1 = self.xp.angle(self.xp.conjugate(self.IRefx) * Ix) * self.alpha
        dw2 = self.xp.angle(self.xp.conjugate(self.IRefy) * Iy) * self.alpha

        # Ensure angle values are on the right device
        cos_angle = self.xp.array(self.cropsX.angle['cos'])
        sin_angle = self.xp.array(self.cropsX.angle['sin'])
        
        # Rotate phase differences to original coordinate system
        dwX = cos_angle * dw1 - sin_angle * dw2
        dwY = sin_angle * dw1 + cos_angle * dw2

        # Calculate frequency domain coordinates - already on right device
        kx = self.xx - self.Nx / 2
        ky = self.yy - self.Ny / 2

        # Compute denominator for integration, handling division by zero
        denom = 1j * 2 * self.xp.pi * (kx / self.Nx + 1j * ky / self.Ny)
        mask=self.xp.abs(denom) < 1e-10

        # Integrate phase gradients
        denom[mask]=1
        quotient=(self.fft(dwX) + 1j * self.fft(dwY))/denom
        quotient[mask]=0
        W0 = self.ifft(quotient)

        # Return real part scaled by pixel size and Z factor
        result = self.xp.real(W0) * self.p / self.Z
        
        # Return result on CPU
        if self.gpu:
            return result.get()
        return result

def get_phase_cmap():
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

phase_cmap = get_phase_cmap()