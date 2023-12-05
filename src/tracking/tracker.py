import cv2
import numpy as np


class Tracker:
    def __init__(self, frame_size):
        '''
            Параметры для выполения метода Lucas-Kanade
            1. winSize: Размер окна, используемого для нахождения оптического потока.
                Это окно смещается от одного кадра к другому для обнаружения движения.
            2. maxLevel: Максимальный уровень пирамиды, используемой для нахождения оптического потока.
                Это позволяет выполнить многоуровневое выравнивание оптического потока для улучшения точности.
            3. criteria: Критерий останова для итеративного алгоритма оптического потока. 
                Это обычно комбинация из максимального количества итераций и минимального значения, используемого для оценки изменения.
        '''
        self.__lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.__frame_size = frame_size
        self.__mask = np.zeros(self.__frame_size)
    
    def track(self, frame_gray_init, frame_gray_cur, object_points):
        # Lucas-Kanade Optical Flow, using OpenCV
        # return the mask with vectors from old to new position 
        old_points = np.array(object_points, dtype=np.float32)
        new_points, status, err = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray_cur, old_points, None, **self.__lk_params)
        
        for idx, point in enumerate(new_points):
            old = tuple(map(int, old_points[idx]))
            new = tuple(map(int, point))
            self.__mask = cv2.line(self.__mask, old, new, color=(255,255,0), thickness=1)
        return self.__mask.astype('uint8')