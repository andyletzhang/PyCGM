import numpy as np
import pyfftw

class FFT_Processor:
    def __init__(
        self, shape, dtype=np.complex128, planning_effort="FFTW_MEASURE", threads=4
    ):
        self.shape = shape
        self.dtype = dtype
        self.num_threads = threads
        self.planning_effort = planning_effort

        self.fft_input = pyfftw.empty_aligned(shape, dtype=dtype)
        self.fft_output = pyfftw.empty_aligned(shape, dtype=dtype)

        self.ifft_input = pyfftw.empty_aligned(shape, dtype=dtype)
        self.ifft_output = pyfftw.empty_aligned(shape, dtype=dtype)

        self.fft_object = pyfftw.FFTW(
            self.fft_input,
            self.fft_output,
            axes=(0, 1),
            direction="FFTW_FORWARD",
            flags=(planning_effort,),
            threads=self.num_threads,
        )

        self.ifft_object = pyfftw.FFTW(
            self.ifft_input,
            self.ifft_output,
            axes=(0, 1),
            direction="FFTW_BACKWARD",
            flags=(planning_effort,),
            threads=self.num_threads,
        )

    def fft(self, input_array):
        """
        Computes fftshift(fft(input_array)) efficiently.

        Parameters:
        -----------
        input_array : numpy.ndarray
            Input array with shape and dtype matching the initialized parameters

        Returns:
        --------
        numpy.ndarray
            The FFT result after shifting
        """
        # Copy input to aligned array
        np.copyto(self.fft_input, input_array)

        # Execute the FFT
        self.fft_object()

        # Shift the result
        return np.fft.fftshift(self.fft_output).copy()

    def ifft(self, input_array):
        """
        Computes ifft(ifftshift(input_array)) efficiently.

        Parameters:
        -----------
        input_array : numpy.ndarray
            Input array with shape matching the initialized parameters

        Returns:
        --------
        numpy.ndarray
            The IFFT result
        """
        # Copy and shift input to aligned array
        np.copyto(self.ifft_input, np.fft.ifftshift(input_array))

        # Execute the IFFT
        self.ifft_object()

        # Return the result
        return self.ifft_output.copy()

    def cleanup(self):
        """
        Clean up the FFTW objects to free memory.
        """
        self.fft_object = None
        self.ifft_object = None
        self.fft_input = None
        self.fft_output = None
        self.ifft_input = None
        self.ifft_output = None