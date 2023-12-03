import cv2
import numpy as np
import sys


class MotionDetectionCV:
    def __init__(self, video_stream):
        self.__video_stream = video_stream
        
        self.height = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.width = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.__min_area = 500
        self.__blur_kernel_size = (5,5)
        self.__computing_frames_count = 1

    def static_video_detect(self):
        frame1 = self.__video_stream.read()
        
        frame_count = 0
        while self.__video_stream.isOpened():
            is_success, frame = self.__video_stream.read()
            if is_success:
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame_count % self.__computing_frames_count == 0 or frame_count == 1:
                    processed_frames = []
                    
                # difference between current frame and background
                difference = cv2.absdiff(gray, background)
                difference = cv2.GaussianBlur(difference, self.__blur_kernel_size, 0)
                _, threshold = cv2.threshold(difference, 25, 255 ,cv2.THRESH_BINARY)
                dilate_frame = cv2.dilate(threshold, None, iterations=2)
                processed_frames.append(dilate_frame)
                
                # get needed count of frames
                if len(processed_frames) == self.__computing_frames_count:
                    sum_frames = (sum(processed_frames)/self.__computing_frames_count).astype('uint8')
                    
                    contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) < self.__min_area:
                            continue
                    
                        (x,y,w,h) = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                        
                    cv2.imshow('video', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            else:
                break
        
        self.__video_stream.release()
        cv2.destroyAllWindows()
    
    def detect(self):
        is_success, frame1 = self.__video_stream.read()
        is_success, frame2 = self.__video_stream.read()

        while is_success and self.__video_stream.isOpened():
            gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
            delta = cv2.absdiff(gray_frame1, gray_frame2)
            delta = cv2.GaussianBlur(delta, self.__blur_kernel_size, 0)
            
            _, threshold = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(threshold, None, iterations=3)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
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