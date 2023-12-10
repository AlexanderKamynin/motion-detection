import numpy as np
from scipy.signal import convolve2d

class GaussianBlur:
    def __init__(self, kernel_size, sigma=0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        self.kernel = np.zeros(kernel_size**2).reshape((kernel_size, kernel_size))
        self.__computeSigma()
        self.__computeKernel()
    
    def blur_image(self, image):
        blurred_img = convolve2d(image, self.kernel).astype('uint8')
        
        return blurred_img
    
    @staticmethod
    def gauss_func(sigma, x, y):
        gaussian = np.exp(-(x**2 + y**2) / (2*sigma**2)) / (2*np.pi*sigma**2)
        return gaussian
    
    def __computeSigma(self):
        if self.sigma <= 0:
            self.sigma = 0.3*((self.kernel_size - 1)*0.5 - 1) + 0.8
            
    def __computeKernel(self):
        radius = self.kernel_size // 2
        for y in range(self.kernel_size):
            for x in range(self.kernel_size):
                self.kernel[y][x] = GaussianBlur.gauss_func(self.sigma, x - radius, y - radius)
        
        kernel_sum = self.kernel.sum()
        for i in range(self.kernel_size):
            self.kernel[i] /= kernel_sum
            
            
if __name__ == '__main__':
    gaussian_blur = GaussianBlur(kernel_size=5)