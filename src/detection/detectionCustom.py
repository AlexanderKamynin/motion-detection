import cv2
import numpy as np
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
    
        self.__min_area = 125
        self.__blur_kernel_size = 5
        self.__gaussian_blur = GaussianBlur(self.__blur_kernel_size)
        self.__morphology = MorphologyOperations()
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
                if frame_count > self.__max_frames:
                    processed_frames.pop(0)
                    
                # converting frames to gray
                gray_frame1 = MotionDetectionCustom.convertRGBtoGray(frame1)
                gray_frame2 = MotionDetectionCustom.convertRGBtoGray(frame2)
                
                difference = abs(gray_frame1 - gray_frame2).astype('uint8')
                # blur
                difference = self.__gaussian_blur.blur_image(difference)
                # threshold
                threshold = ((difference > self.__threshold) * 255).astype('uint8')
                dilated = self.__morphology.dilate(threshold)
                
                processed_frames.append(dilated)
                accumulate_frame = np.mean(processed_frames, axis=0).astype('uint8')
                
                cv2.imshow("origin", accumulate_frame)
                contours = self.__find_contours(accumulate_frame)
                for contour in contours:
                    left_up, right_down = contour
                    cv2.rectangle(accumulate_frame, left_up, right_down, 255, 2)
                
                cv2.imshow("contour", accumulate_frame)
                cv2.waitKey(0)
                
                cv2.imshow("video", difference)
                frame1 = frame2
                is_success, frame2 = self.__video_stream.read()
                
                if cv2.waitKey(15) & 0xFF == ord('q'):
                    break
            else:
                break
        
        self.__video_stream.release()
        cv2.destroyAllWindows()
        
    def __find_contours(self, image):
        h, w = image.shape[:2]
        visited = np.zeros((h,w), dtype=bool)
        contours = []
        for row in range(h):
            for col in range(w):
                if not visited[row][col] and image[row][col]:
                    queue = []
                    
                    queue.append((col, row))
                    object_contour = [[col,row], [col,row]]
                    
                    while queue:
                        x, y = queue.pop(0)
                        visited[y, x] = True
                        
                        # left
                        if x - 1 >= 0 and image[y, x-1] >= 170 and not visited[y, x-1]:
                            queue.append((x-1,y))
                            visited[y, x-1] = True
                            # update left corner
                            if x - 1 < object_contour[0][0]:
                                object_contour[0][0] = x - 1
                                
                        # up
                        if y - 1 >= 0 and image[y-1, x] >= 170 and not visited[y-1, x]:
                            queue.append((x,y-1))
                            visited[y-1, x] = True
                            if y - 1 < object_contour[0][1]:
                                object_contour[0][1] = y - 1
                                
                        # right
                        if y + 1 < h and image[y+1, x] >= 170 and not visited[y+1, x]:
                            queue.append((x,y+1))
                            visited[y+1, x] = True
                            if y + 1 > object_contour[1][1]:
                                object_contour[1][1] = y + 1
                        # down
                        if x + 1 < w and image[y, x+1] >= 170 and not visited[y, x+1]:
                            queue.append((x+1,y))
                            visited[y, x+1] = True
                            if x + 1 > object_contour[1][0]:
                                object_contour[1][0] = x + 1
                    
                    # check that area is above that min_area
                    if (object_contour[1][0] - object_contour[0][0]) * (object_contour[1][1] - object_contour[0][1]) >= self.__min_area:
                        contours.append(object_contour)
                        print(f'Now I found {object_contour}')
        print(f'Contour numbers is equal to {len(contours)}')
        
        return contours

    @staticmethod             
    def convertRGBtoGray(image):
        # works faster without pre-conversion
        #float_img = np.array([[pixel / 255 for pixel in row] for row in image])

        # we don't need to clip values, because in worst case have 0.299*255 + 0.587*255 + 0.114*255 = 255 <= 255
        gray_img = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
        gray_img = gray_img.astype('int')
        return gray_img

if __name__ == '__main__':
    video_stream = cv2.VideoCapture('../videos/test1.mp4')
    md = MotionDetectionCustom(video_stream)
    md.detect()