import numpy as np


class MorphologyOperations:
    def __init__(self, dilatation_kernel=None, erosion_kernel=None):
        self.__dilatation_kernel = dilatation_kernel
        self.__erosion_kernel = erosion_kernel
        if not dilatation_kernel:
            self.__dilatation_kernel = np.ones((14,14))
        if not erosion_kernel:
            self.__erosion_kernel = np.ones((3,3))
        
    def dilate(self, image, resize_coef = 4):
        x = self.__dilatation_kernel.shape[0] // 2
        y = self.__dilatation_kernel.shape[1] // 2
        processed_image = np.zeros(np.array(image.shape) // resize_coef, dtype=np.uint8)
        for row in range(y, image.shape[0] - y, resize_coef):
            for col in range(x, image.shape[1] - x, resize_coef):
                local_window = image[row - y : row + y + 1, col - x: col + x + 1]
                processed_image[row // resize_coef][col // resize_coef] = np.max(local_window)
                
        return processed_image
    
    def erosion(self, image, resize_coef = 1):
        x = self.__erosion_kernel.shape[0] // 2
        y = self.__erosion_kernel.shape[1] // 2
        processed_image = np.zeros(np.array(image.shape) // resize_coef, dtype=np.uint8)
        for row in range(y, image.shape[0] - y, resize_coef):
            for col in range(x, image.shape[1] - x, resize_coef):
                local_window = image[row - y : row + y + 1, col - x: col + x + 1]
                processed_image[row // resize_coef][col // resize_coef] = np.min(local_window)

        return processed_image