import cv2
import numpy as np

class MotionDetectionCV:
    def __init__(self, video_stream):
        self.__video_stream = video_stream
        
        self.__min_area = 250
        self.__blur_kernel_size = (5,5)
        
    def get_background(self):
        # get randomly frames for calculating median 
        frame_indices = self.__video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=75)
        
        frames = []
        for idx in frame_indices:
            self.__video_stream.set(cv2.CAP_PROP_POS_FRAMES, idx)
            is_success, frame = self.__video_stream.read()
            frames.append(frame)
        
        background_frame = np.median(frames, axis=0).astype(np.uint8)
        
        return background_frame

    def static_video_detect(self):
        background = self.get_background()
        cv2.imshow('image', background)
        cv2.waitKey(0)
    
    def detect(self):
        is_success, frame1 = self.__video_stream.read()
        is_success, frame2 = self.__video_stream.read()

        while is_success and self.__video_stream.isOpened():
            gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
            delta = cv2.absdiff(gray_frame1, gray_frame2)
            delta = cv2.GaussianBlur(delta, self.__blur_kernel_size, 0)
            
            is_success, threshold = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(threshold, None, iterations=3)
            
            contours, is_success = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < self.__min_area:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow("video", frame1)
            frame1 = frame2
            is_success, frame2 = self.__video_stream.read()
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
        self.__video_stream.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    video_stream = cv2.VideoCapture('../../videos/default.mp4')
    md = MotionDetectionCV(video_stream)
    md.static_video_detect()