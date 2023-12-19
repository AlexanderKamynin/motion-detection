# import dependencies
import cv2
import numpy as np
from collections import deque
import typing
import os
import sys

# import project components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from components.geometry import Geometry


class MotionDetectionCV:
    def __init__(self,
                min_area: int = 500,
                blur_kernel_size: typing.Tuple[int, int] = (5,5),
                threshold: int = 25,
                max_frames: int = 4):
        """
        Initializes an instance of the MotionDetector class for motion detection on a video stream using OpenCV library
        
        :param min_area: Minimum area of an object to be considered as motion. Default: 500.
        :type min_area: int
        
        :param blur_kernel_size: Size of the kernel for image blurring. Default: (5, 5).
        :type blur_kernel_size: tuple[int, int]

        :param threshold: Threshold value for motion detection. Default: 25.
        :type threshold: int

        :param max_frames: Maximum number of frames stored for averaging past frames when defining object contours. Default: 4.
        :type max_frames: int
        
        :ivar __min_area: Minimum area for the detected motion.
        :ivar __blur_kernel_size: Kernel size for image blurring.
        :ivar __threshold: Threshold value for motion detection.
        :ivar __max_frames: Maximum number of frames for analysis.
        :ivar __frame_count: Count of processed frames.
        :ivar __processed_frames: Deque for storing processed frames.
        """
        # detection parameters
        self.__min_area = min_area
        self.__blur_kernel_size = blur_kernel_size
        self.__threshold = threshold
        self.__max_frames = max_frames
        
        # processed frame parameters
        self.__frame_count = 0
        self.__processed_frames = deque()

    def detect(self, old_gray_frame: np.ndarray, new_gray_frame: np.ndarray) -> typing.List[typing.List[tuple]]:
        """
        Detects motion in a video stream based on the difference between two grayscale frames
        
        :param old_gray_frame: Grayscale previous frame
        :type old_gray_frame: numpy.ndarray
        
        :param new_gray_frame: Grayscale current frame
        :type new_gray_frame: numpy.ndarray
        
        :return: A list of bounding rectangles [(x1, y1), (x2, y2)] representing the detected motion areas.
        :rtype: List[List[tuple]]
        
        """
        # when frame count is more than max computing frames, using in detect - update processed frames
        self.__frame_count += 1
        if self.__frame_count > self.__max_frames:
            self.__processed_frames.popleft()
                        
        difference = cv2.absdiff(old_gray_frame, new_gray_frame)
        difference = cv2.GaussianBlur(difference, self.__blur_kernel_size, 0)
        _, threshold = cv2.threshold(difference, self.__threshold, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(threshold, None, iterations=3)
        # cv2.imshow("dilated", dilated)
        self.__processed_frames.append(dilated)
                
        accumulate_frame = np.mean(self.__processed_frames, axis=0).astype('uint8')
        # cv2.RETR_EXTERNAL provide deleting all inner (daughter) contours
        contours, hierarchy = cv2.findContours(accumulate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # contouring area around the detected objects    
        bounded_rectangles = []
        for contour in contours:
            if cv2.contourArea(contour) < self.__min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            bounded_rectangles.append([(x, y), (x+w, y+h)])
                
        bounded_rectangles = self.__delete_inner_rectangles(bounded_rectangles)

        return bounded_rectangles
        
    def __delete_inner_rectangles(self, bounded_rectangles: typing.List[typing.List[tuple]]) -> typing.List[typing.List[tuple]]: 
        """
        Deletes inner rectangles from a list of all bounding rectangles.
        
        :param bounded_rectangles: A list of bounding rectangles [(x1, y1), (x2, y2)].
        :type bounded_rectangles: List[List[tuple]]
        
        :return: A filtered list of bounding rectangles without inner rectangles.
        :rtype: List[List[tuple]]
        """
        if not len(bounded_rectangles):
            return
        
        # it's more likely that smaller rectangles will fall into the other ones
        rect_count = len(bounded_rectangles)
        bounded_rectangles = sorted(bounded_rectangles, key=lambda rect: (rect[1][0] - rect[0][0])*(rect[1][1] - rect[0][1]), reverse=True) # sort by rectangles area
        
        is_inner = [True] * rect_count
        for i in range(rect_count):
            if is_inner[i]:
                for j in range(i+1, rect_count):
                    if is_inner[j] and Geometry.rect_contain_another(bounded_rectangles[i], bounded_rectangles[j]):
                        is_inner[j] = False
        
        bounded_rectangles = [bounded_rectangles[i] for i in range(rect_count) if is_inner[i]]
                        
        return bounded_rectangles
