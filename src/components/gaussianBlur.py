import numpy as np


class GaussianBlur:
    def __init__(self, kernel_size, sigmaX=0, sigmaY=0):
        self.kernel_size = kernel_size
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY
        
        self.kernel = np.zeros(np.prod(self.kernel_size)).reshape(self.kernel_size)
        self.__computeSigma()
        self.__computeKernel()
        print(self.kernel)
    
    def GaussianBlur(self, image):
        pass
    
    @staticmethod
    def gauss_func(sigma, x, y):
        gaussian = np.exp((-x**2 - y**2) / (2*sigma**2)) / (2*np.pi*sigma**2)
        return gaussian
    
    def __computeSigma(self):
        if self.sigmaX == 0:
            self.sigmaX = 0.3*((self.kernel_size[0]-1)*0.5 - 1) + 0.8
            
    def __computeKernel(self):
        radius = (self.kernel_size[0] - 1)//2
        for y in range(self.kernel_size[0]):
            for x in range(self.kernel_size[1]):
                self.kernel[y][x] = GaussianBlur.gauss_func(self.sigmaX, x - radius, y - radius)
        
        kernel_sum = self.kernel.sum()
        for i in range(self.kernel_size[0]):
            self.kernel[i] /= kernel_sum