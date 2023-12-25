# import dependencies
import numpy as np


class MorphologyOperations:
    def __init__(
        self, dilatation_kernel: tuple = None, erosion_kernel: tuple = None
    ) -> None:
        """
        Initialize the object with the specified dilatation and erosion kernels.

        Parameters
        ----------
        dilatation_kernel : tuple, optional
            The kernel for dilatation operation. Defaults to a 14x14 matrix of ones if not provided.

        erosion_kernel : tuple, optional
            The kernel for erosion operation. Defaults to a 3x3 matrix of ones if not provided.

        Returns
        -------
        None
        """
        self.__dilatation_kernel = dilatation_kernel
        self.__erosion_kernel = erosion_kernel
        if not dilatation_kernel:
            self.__dilatation_kernel = np.ones((14, 14))
        if not erosion_kernel:
            self.__erosion_kernel = np.ones((3, 3))

    def dilate(self, image, resize_coef: int = 1) -> np.ndarray:
        """
        Apply dilatation operation to the input image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image.

        resize_coef : int, optional
            Coefficient for resizing the output image. Used to reduce the output size and optimize calculations. Default is 1.

        Returns
        -------
        numpy.ndarray
            The dilated image.
        """
        x = self.__dilatation_kernel.shape[0] // 2
        y = self.__dilatation_kernel.shape[1] // 2
        processed_image = np.zeros(np.array(image.shape) // resize_coef, dtype=np.uint8)
        for row in range(y, image.shape[0] - y, resize_coef):
            for col in range(x, image.shape[1] - x, resize_coef):
                local_window = image[row - y : row + y + 1, col - x : col + x + 1]
                processed_image[row // resize_coef][col // resize_coef] = np.max(
                    local_window
                )

        return processed_image

    def erosion(self, image, resize_coef: int = 1) -> np.ndarray:
        """
        Apply erosion operation to the input image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image.

        resize_coef : int, optional
            Coefficient for resizing the output image. Used to reduce the output size and optimize calculations. Default is 1.

        Returns
        -------
        numpy.ndarray
            The erosed image.
        """
        x = self.__erosion_kernel.shape[0] // 2
        y = self.__erosion_kernel.shape[1] // 2
        processed_image = np.zeros(np.array(image.shape) // resize_coef, dtype=np.uint8)
        for row in range(y, image.shape[0] - y, resize_coef):
            for col in range(x, image.shape[1] - x, resize_coef):
                local_window = image[row - y : row + y + 1, col - x : col + x + 1]
                processed_image[row // resize_coef][col // resize_coef] = np.min(
                    local_window
                )

        return processed_image
