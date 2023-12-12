import numpy as np


class Dilatation:
    def __init__(self, kernel=None):
        self.__kernel = kernel
        if not kernel:
            self.__kernel = np.ones((3,3))
        
    def dilate(self, image, iterations):
        height, width = image.shape
        padding = self.__kernel.shape[0] // 2
        
        result = np.copy(image)
        for _ in range(iterations):
            # Создаем массив с отступами
            padded_image = np.pad(image, padding, mode='constant', constant_values=0)

            # Используем свертку для эффективного применения ядра
            dilated_image = np.maximum.reduce([
                padded_image[i:i+self.__kernel.shape[0], j:j+self.__kernel.shape[1]] * self.__kernel
                for i in range(image.shape[0])
                for j in range(image.shape[1])
            ])

            result = np.maximum(result, dilated_image)
                    
        return result
    