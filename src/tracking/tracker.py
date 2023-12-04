import cv2
import numpy as np


class Tracker:
    def __init__(self):
        '''
            1. maxCorners: Максимальное количество углов, которые должны быть обнаружены.
                Если обнаружено больше углов, то будут выбраны те, которые имеют самое высокое качество.
            2. qualityLevel: Пороговое значение качества, используемое для отбора углов
            3. minDistance: Минимальное евклидово расстояние между обнаруженными углами
            4. blockSize: Размер окна, используемого для вычисления углов
        '''
        self.__feature_params = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 7,
            blockSize = 7
        )
        
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
    
    def track(self, origin_frame1, gray_frame1, origin_frame2, gray_frame2):
        # Lucas-Kanade Optical Flow, using OpenCV
        p0 = cv2.goodFeaturesToTrack(gray_frame1, mask=None, **self.__feature_params)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_frame1, gray_frame2, p0, None, **self.__lk_params)
        
        mask = np.zeros_like(origin_frame1)
        good_new = p1[st==1].astype(int)
        good_old = p0[st==1].astype(int)
        
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a,b), (c,d), (0,0,255), 2)
            origin_frame2 = cv2.circle(origin_frame2, (a,b), 5, (255,0,0), -1)
        img = cv2.add(origin_frame2, mask)
        
        cv2.imshow('optic', img)
        return