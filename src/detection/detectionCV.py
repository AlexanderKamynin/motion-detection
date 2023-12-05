import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tracking.tracker import Tracker

class MotionDetectionCV:
    
    def __init__(self, video_stream):
        self.__video_stream = video_stream
        
        self.__height = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__width = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__channel_numbers = 0
        
        self.__tracker = None
    
        self.__min_area = 500
        self.__blur_kernel_size = (5,5)
        self.__threshold = 25
        self.__max_frames = 4

    def detect(self):
        is_success, frame1 = self.__video_stream.read()
        
        self.__channel_numbers = frame1.shape[-1]
        self.__tracker = Tracker((self.__width, self.__height, self.__channel_numbers))
        
        frame_count = 0
        processed_frames = []

        while self.__video_stream.isOpened():
            is_success, frame2 = self.__video_stream.read()
            if is_success:
                frame_count += 1
                # when frame count is more than max computing frames, using in detect - update processed frames
                # just delete the first element from history
                if frame_count > self.__max_frames:
                    processed_frames.pop(0)
                
                gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
                difference = cv2.absdiff(gray_frame1, gray_frame2)
                difference = cv2.GaussianBlur(difference, self.__blur_kernel_size, 0)
                
                _, threshold = cv2.threshold(difference, self.__threshold, 255, cv2.THRESH_BINARY)
                
                dilated = cv2.dilate(threshold, None, iterations=3)
                processed_frames.append(dilated)
                
                accumulate_frame = np.mean(processed_frames, axis=0).astype('uint8')
                # cv2.RETR_EXTERNAL provide deleting all inner (daughter) contours
                contours, hierarchy = cv2.findContours(accumulate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                bounded_rectangles = []
                for contour in contours:
                    if cv2.contourArea(contour) < self.__min_area:
                        continue
                    (x, y, w, h) = cv2.boundingRect(contour)
                    
                    bounded_rectangles.append([(x, y), (x+w, y+h)])
                    
                bounded_rectangles = self.__delete_inner_rectangles(bounded_rectangles)
                
                object_points = []
                # draw all rect
                if bounded_rectangles:
                    for rectangle in bounded_rectangles:
                        cv2.rectangle(frame1, (rectangle[0][0], rectangle[0][1]), (rectangle[1][0], rectangle[1][1]), (0, 255, 0), 2)
                        rectangle_center = MotionDetectionCV.get_center_coords(rectangle)
                        cv2.circle(frame1, rectangle_center, 3, (255, 0, 0), 1)
                        object_points.append(rectangle_center)
                
                mask = self.__tracker.track(gray_frame1, gray_frame2, object_points)
                
                result = cv2.add(frame1, mask)
                cv2.imshow("video", result)
                frame1 = frame2
                is_success, frame2 = self.__video_stream.read()
                
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
            else:
                break
            
        self.__video_stream.release()
        cv2.destroyAllWindows()
        
    def __delete_inner_rectangles(self, bounded_rectangles):
        if not len(bounded_rectangles):
            return
        
        # it's more likely that smaller rectangles will fall into the other ones
        rect_number = len(bounded_rectangles)
        bounded_rectangles = sorted(bounded_rectangles, key=lambda rect: (rect[1][0] - rect[0][0])*(rect[1][1] - rect[0][1]), reverse=True) # sort by rectangles area
        
        is_inner = [True] * rect_number
        for i in range(rect_number):
            if is_inner[i]:
                for j in range(i+1, rect_number):
                    if is_inner[j] and MotionDetectionCV.is_contain(bounded_rectangles[i], bounded_rectangles[j]):
                        is_inner[j] = False
        
        bounded_rectangles = [bounded_rectangles[i] for i in range(rect_number) if is_inner[i]]
                        
        return bounded_rectangles
                        
    @staticmethod
    def is_contain(rect1, rect2):
        (top_x1, top_y1), (down_x1, down_y1) = rect1
        (top_x2, top_y2), (down_x2, down_y2) = rect2

        if top_x1 >= top_x2 and down_x1 <= down_x2 and top_y1 >= top_y2 and down_y1 <= down_y2:
            return True
        # Проверка, что rect2 полностью содержится в rect1
        elif top_x2 >= top_x1 and down_x2 <= down_x1 and top_y2 >= top_y1 and down_y2 <= down_y1:
            return True
        return False
    
    @staticmethod
    def get_center_coords(rect):
        # tuple (x; y)
        x = (rect[1][0] + rect[0][0]) // 2
        y = (rect[1][1] + rect[0][1]) // 2
        return (x, y)


if __name__ == '__main__':
    video_stream = cv2.VideoCapture('../videos/test1.mp4')
    # video_stream = cv.VideoCapture('http://192.168.217.103/mjpg/video.mjpg')
    md = MotionDetectionCV(video_stream)
    md.detect()