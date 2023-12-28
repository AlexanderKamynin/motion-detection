# import dependencies
import numpy as np
from scipy.signal import convolve2d


class GaussianBlur:
    def __init__(self, kernel_size: int, sigma: int = 0) -> None:
        """
        Initialize a custom filter with a specified kernel size and optional sigma.

        Parameters
        ----------
        kernel_size : int
            Size of the square kernel (e.g., 3 for a 3x3 kernel).

        sigma : int, optional, default: 0
            Standard deviation for Gaussian smoothing. If not provided, sigma will be computed optional
        """
        self.__kernel_size = kernel_size
        self.__sigma = sigma

        self.__kernel = np.zeros(kernel_size**2).reshape((kernel_size, kernel_size))
        self.__computeSigma()
        self.__computeKernel()

    def blur_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the blur filter for the input image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image to be blurred.

        Returns
        -------
        numpy.ndarray
            The blurred image after applying.
        """
        blurred_image = convolve2d(image, self.__kernel, mode="same").astype("uint8")

        # blurred_image = np.zeros(image.shape)
        # h, w = image.shape[:2]

        # for row in range(h + 1):
        #     for col in range(w + 1):
        #         if row + self.__kernel_size <= h and col + self.__kernel_size <= w:
        #             batch = image[
        #                 row : row + self.__kernel_size, col : col + self.__kernel_size
        #             ]
        #             blurred_image[row, col] = (batch * self.__kernel).sum()
        # import cv2

        # blurred_image = blurred_image.astype("uint8")
        # cv2.imshow("blurred", blurred_image)

        return blurred_image

    @staticmethod
    def gauss_func(sigma: float, x: int, y: int) -> float:
        """
        Compute the value of a 2D Gaussian function.

        Parameters
        ----------
        sigma : float
            Standard deviation of the Gaussian function.

        x : float
            X-coordinate.

        y : float
            Y-coordinate.

        Returns
        -------
        float
            The computed value of the Gaussian function at the specified coordinates.
        """
        gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (
            2 * np.pi * sigma**2
        )
        return gaussian

    def __computeSigma(self) -> None:
        """
        Compute the value of sigma for Gaussian smoothing.

        If the provided sigma is less than or equal to 0, it is calculated based on a formula from OpenCV documentation.

        Returns
        -------
        None
        """
        if self.__sigma <= 0:
            self.__sigma = 0.3 * ((self.__kernel_size - 1) * 0.5 - 1) + 0.8

    def __computeKernel(self) -> None:
        """
        Compute the kernel matrix using a Gaussian function.

        Returns
        -------
        None
        """
        radius = self.__kernel_size // 2
        for y in range(self.__kernel_size):
            for x in range(self.__kernel_size):
                self.__kernel[y][x] = GaussianBlur.gauss_func(
                    self.__sigma, x - radius, y - radius
                )

        kernel_sum = self.__kernel.sum()
        for i in range(self.__kernel_size):
            self.__kernel[i] /= kernel_sum


if __name__ == "__main__":
    gaussian_blur = GaussianBlur(kernel_size=5)
