import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class MotionDetectionCustom:
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
        # skip tracking until it's not to be done
        
        frame_count = 0
        processed_frames = []
        
        while self.__video_stream.isOpened():
            is_success, frame2 = self.__video_stream.read()
            if is_success:
                frame_count += 1
                # when frame count is more than max computing frames, using in detect - update processed frames
                # just delete the first element from history
                # if frame_count > self.__max_frames:
                #     processed_frames.pop(0)
                    
                # converting frames to gray
                gray_frame1 = MotionDetectionCustom.convertRGBtoGray(frame1)
                gray_frame2 = MotionDetectionCustom.convertRGBtoGray(frame2)
                difference = abs(gray_frame1 - gray_frame2).astype('uint8')
                
                cv2.imshow("video", difference)
                frame1 = frame2
                is_success, frame2 = self.__video_stream.read()
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        
        self.__video_stream.release()
        cv2.destroyAllWindows()

    @staticmethod             
    def convertRGBtoGray(image):
        # works faster without pre-conversion
        #float_img = np.array([[pixel / 255 for pixel in row] for row in image])

        # we don't need to clip values, because in worst case have 0.299*255 + 0.587*255 + 0.114*255 = 255 <= 255
        gray_img = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
        gray_img = gray_img.astype('int')
        return gray_img
    
    def GaussianBlur(self, image, kernel_size):
        pass

if __name__ == '__main__':
    video_stream = cv2.VideoCapture('../videos/test1.mp4')
    md = MotionDetectionCustom(video_stream)
    md.detect()