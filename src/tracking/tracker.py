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
        self.__max_point = 30
        self.__colors = np.random.randint(0, 255, (15, 3))
        self.__points_history = []
    
    def track(self, frame_gray_init, frame_gray_cur, object_points):
        # Lucas-Kanade Optical Flow, using OpenCV
        # return the mask with vectors from old to new position 
        old_points = np.array(object_points, dtype=np.float32)
        new_points, _, _ = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray_cur, old_points, None, **self.__lk_params)
        
        #save points
        self.__points_history.append({'old_points': old_points, 'new_points': new_points})
        
        for idx, point in enumerate(new_points):
            old = tuple(map(int, old_points[idx]))
            new = tuple(map(int, point))
            self.__mask = cv2.line(self.__mask, old, new, self.__colors[idx].tolist(), thickness=2)
            
        if len(self.__points_history) > self.__max_point:
            self.__clear_last_points()
            
        return self.__mask.astype('uint8')
    
    def __clear_last_points(self):
        last_points = self.__points_history.pop(0)
        
        old = last_points['old_points']
        new = last_points['new_points']
        
        # set black on the mask
        for idx, point in enumerate(old):
            cv2.line(self.__mask, tuple(map(int ,point)), tuple(map(int, new[idx])), color=(0,0,0))