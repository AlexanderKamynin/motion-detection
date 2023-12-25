# import dependencies
import numpy as np


class ImageProcessingUtils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def convertRGBtoGray(image: np.ndarray) -> np.ndarray:
        """
        Convert an RGB image to grayscale.

        This method uses the luminosity method to convert RGB to grayscale.

        Parameters
        ----------
        image : np.ndarray
            Input RGB image.

        Returns
        -------
        np.ndarray
            Grayscale image.
        """
        # works faster without pre-conversion
        # float_img = np.array([[pixel / 255 for pixel in row] for row in image])

        # we don't need to clip values, because in worst case have 0.299*255 + 0.587*255 + 0.114*255 = 255 <= 255
        gray_img = (
            0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        )
        gray_img = gray_img.astype("int")
        return gray_img
