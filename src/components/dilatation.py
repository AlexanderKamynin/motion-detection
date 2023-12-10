import numpy as np


class Dilatation:
    def __init__(self, kernel=None):
        self.__kernel = kernel
        if not kernel:
            self.__kernel = np.ones((3,3))
            
        print(self.__kernel)
        
        
    def dilate(image, iterations):
        for _ in range(iterations):
            pass
        #     rows, cols = image.shape
        #     krows, kcols = kernel.shape
            
        #     result = np.zeros((rows, cols)).astype('uint8')
            
        #     anchorX, anchorY = kcols // 2, krows // 2
            
        #     for x in range(rows):
        #         for y in range(cols):
        #             if image[x, y] == 255:
        #                 for i in range(krows):
        #                     for j in range(kcols):
        #                         imgX, imgY = x - anchorX + i, y - anchorY + j
                                
        #                         if 0 <= imgX < rows and 0 <= imgY < cols:
        #                             result[imgX, imgY] = 255
        
        # return result
            
        
if __name__ == '__main__':
    dilatation = Dilatation()
    