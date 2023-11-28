# import dependencies
import cv2

# import project modules
from src.detection.openCV.detectionCV import MotionDetectionCV

class Core:
    def __init__(self):
        self.__video_stream = None
        self.__motion_detection_cv = None
        
    def start(self):
        self.__read_video()
        self.__motion_detection_cv = MotionDetectionCV(self.__video_stream)
        
        self.__motion_detection_cv.detect()
    
    def __read_video(self):
        print('Start read the video...')
        self.__video_stream = cv2.VideoCapture('src/videos/default.mp4')
        if(self.__video_stream == False):
            raise ValueError('Error opening video stream')


if __name__ == '__main__':
    core = Core()
    core.start()
