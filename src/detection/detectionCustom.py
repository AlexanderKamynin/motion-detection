import cv2
import numpy as np
from collections import deque
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from components.gaussianBlur import GaussianBlur
from components.morphology import MorphologyOperations


class MotionDetectionCustom:
    def __init__(self, video_stream):
        self.__video_stream = video_stream
        
        self.__height = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__width = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__channel_numbers = 0
        
        self.__tracker = None
    
        self.__min_area = 50
        self.__blur_kernel_size = 5
        self.__gaussian_blur = GaussianBlur(self.__blur_kernel_size)
        self.__morphology = MorphologyOperations()
        self.__threshold = 25
        self.__bounding_threshold = 200
        self.__max_frames = 1
        
    def detect(self):
        is_success, frame1 = self.__video_stream.read()
        
        self.__channel_numbers = frame1.shape[-1]
        # skip tracking until it's not to be done
        
        frame_count = 0
        processed_frames = deque()
        
        while self.__video_stream.isOpened():
            is_success, frame2 = self.__video_stream.read()
            if is_success:
                frame_count += 1
                # when frame count is more than max computing frames, using in detect - update processed frames
                # just delete the first element from history
                if frame_count > self.__max_frames:
                    processed_frames.popleft()
                    
                # converting frames to gray
                gray_frame1 = MotionDetectionCustom.convertRGBtoGray(frame1)
                gray_frame2 = MotionDetectionCustom.convertRGBtoGray(frame2)
                
                difference = abs(gray_frame1 - gray_frame2).astype('uint8')
                # blur
                difference = self.__gaussian_blur.blur_image(difference)
                # threshold
                threshold = ((difference > self.__threshold) * 255).astype('uint8')
                
                resize_coef = 4
                dilated = self.__morphology.dilate(threshold, resize_coef)
                
                processed_frames.append(dilated)
                accumulate_frame = np.mean(processed_frames, axis=0).astype('uint8')
                
                cv2.imshow("dilated", accumulate_frame)
                contours = self.__find_contours(accumulate_frame)
                for contour in contours:
                    left_up, right_down = contour
                    left_up = (left_up[0] * 4, left_up[1] * resize_coef)
                    right_down = (right_down[0] * 4, right_down[1] * resize_coef)
                    cv2.rectangle(frame1, left_up, right_down, (0, 255, 0), 2)
                
                cv2.imshow("difference", difference)
                cv2.imshow("video", frame1)
                frame1 = frame2
                is_success, frame2 = self.__video_stream.read()
                
                if cv2.waitKey(15) & 0xFF == ord('q'):
                    break
            else:
                break
        
        self.__video_stream.release()
        cv2.destroyAllWindows()
        
    def __find_contours(self, image):
        #start_time = time.perf_counter()
        h, w = image.shape[:2]
        visited = np.zeros((h,w), dtype=bool)
        in_image = lambda nx, ny: 0 <= nx < w and 0 <= ny < h
        contours = []
        queue = deque()
        
        def update_contour(object_contour, x, y):
            object_contour[0][0] = min(object_contour[0][0], x)
            object_contour[0][1] = min(object_contour[0][1], y)
            object_contour[1][0] = max(object_contour[1][0], x)
            object_contour[1][1] = max(object_contour[1][1], y)
        
        #valid = np.where(image >= self.__bounding_threshold)
        #for row, col in zip(valid[0], valid[1]):
        
        for row in range(0, h, 2):
            for col in range(0, w, 2):
                if not visited[row][col] and image[row][col] and not MotionDetectionCustom.dot_inside_contours(row, col, contours):                    
                    queue.append((col, row))
                    object_contour = [[col,row], [col,row]]
                    
                    while queue:
                        x, y = queue.popleft()
                        visited[y, x] = True
                        
                        neighbors = np.array([
                            (x-1, y), # left
                            (x, y-1), # up
                            (x+1, y), # right
                            (x, y+1) # down
                        ])
                        
                        for neighbor in neighbors:
                            nx, ny = neighbor
                            if not visited[ny, nx] and in_image(nx, ny) and image[ny, nx] >= self.__bounding_threshold:
                                queue.append((nx, ny))
                                visited[ny, nx] = True
                                # update object_contour
                                update_contour(object_contour, nx, ny)
                                
                    
                    # check that area is above that min_area
                    if (object_contour[1][0] - object_contour[0][0]) * (object_contour[1][1] - object_contour[0][1]) >= self.__min_area:
                        contours.append(object_contour)
                                #print(f'Now I found {object_contour}')
                                
        #print(f'Contour numbers is equal to {len(contours)}')
        #print("{:g} s".format(time.perf_counter() - start_time))
        return contours

    @staticmethod             
    def convertRGBtoGray(image):
        # works faster without pre-conversion
        #float_img = np.array([[pixel / 255 for pixel in row] for row in image])

        # we don't need to clip values, because in worst case have 0.299*255 + 0.587*255 + 0.114*255 = 255 <= 255
        gray_img = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
        gray_img = gray_img.astype('int')
        return gray_img
    
    @staticmethod
    def dot_inside_contours(y, x, contours):
        for rect in contours:
            if rect[0][0] <= x <= rect[1][0] and rect[0][1] <= y <= rect[1][1]:
                return True
        return False

if __name__ == '__main__':
    video_stream = cv2.VideoCapture('../videos/test1.mp4')
    md = MotionDetectionCustom(video_stream)
    md.detect()