import cv2

class MotionDetectionCV:
    def __init__(self, video_stream):
        self.__video_stream = video_stream
        
        self.__min_area = 250
    
    def detect(self):
        is_success, frame1 = self.__video_stream.read()
        is_success, frame2 = self.__video_stream.read()

        while is_success and self.__video_stream.isOpened():
            gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
            delta = cv2.absdiff(gray_frame1, gray_frame2)
            delta = cv2.GaussianBlur(delta, (5, 5), 0)
            
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
    pass