import numpy as np


class Dilatation:
    def __init__(self, kernel=None):
        self.__kernel = kernel
        if not kernel:
            self.__kernel = np.ones((14,14))
        
    def dilate(self, image, iterations):
        x = self.__kernel.shape[0] // 2
        y = self.__kernel.shape[1] // 2
        processed_image = np.zeros(np.array(image.shape) // 4, dtype=np.uint8)
        for row in range(y, image.shape[0] - y, 4):
            for col in range(x, image.shape[1] - x, 4):
                local_window = image[row - y : row + y + 1, col - x: col + x + 1]
                processed_image[row // 4][col // 4] = np.max(local_window)
                
        import cv2
        cv2.imshow('dilatation', processed_image)
        return processed_image