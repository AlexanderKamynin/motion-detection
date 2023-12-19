# import dependencies
import cv2

# import project modules
from src.detection.detectionCV import MotionDetectionCV
from src.tracking.tracker import Tracker
from src.const.constant import *
from src.components.geometry import Geometry


class Core:
    def __init__(self):
        self.__video_stream = None
        self.__height = 0
        self.__width = 0
        self.__motion_detection = None
        self.__tracker = None
        
    def start(self):
        self.__read_video()
        
        self.__height = int(self.__video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__width = int(self.__video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        self.__run_detection()
        
    def __run_detection(self):
        print('Start detection...')
        is_success, frame1 = self.__video_stream.read()
        
        # initialization the objects using frame properties
        self.__tracker = Tracker(frame1.shape)
        self.__motion_detection = MotionDetectionCV()
        
        while self.__video_stream.isOpened():
            is_success, frame2 = self.__video_stream.read()
            if is_success:
                gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                bounded_rectangles = self.__motion_detection.detect(gray_frame1, gray_frame2)
                    
                object_points = []
                # draw all rectangles
                if bounded_rectangles:
                    for rectangle in bounded_rectangles:
                        cv2.rectangle(frame1, (rectangle[0][0], rectangle[0][1]), (rectangle[1][0], rectangle[1][1]), (0, 255, 0), 2)
                        rectangle_center = Geometry.get_rect_center(rectangle)
                        cv2.circle(frame1, rectangle_center, 3, (255, 0, 0), 1)
                        object_points.append(rectangle_center)
                
                trajectory_mask = self.__tracker.track(gray_frame1, gray_frame2, object_points)
                
                processed_frame = cv2.add(frame1, trajectory_mask)
                cv2.imshow("video", processed_frame)
                frame1 = frame2
                is_success, frame2 = self.__video_stream.read()
                
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
            else:
                break
            
        self.__video_stream.release()
        cv2.destroyAllWindows()
        print('End detection...')
    
    def __read_video(self):
        print('Start read the video...')
        self.__video_stream = cv2.VideoCapture(VIDEOSTREAM_PATH + 'test1.mp4')
        # self.__video_stream = cv.VideoCapture('http://192.168.217.103/mjpg/video.mjpg')
        if not self.__video_stream:
            raise ValueError('Error opening video stream')
        else:
            print('Video read correctly!')


if __name__ == '__main__':
    core = Core()
    core.start()
