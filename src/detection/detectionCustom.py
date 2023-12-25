# import dependencies
import cv2  # used for log
import numpy as np
from collections import deque
import typing
import sys
import os

# import project components
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from components.gaussianBlur import GaussianBlur
from components.morphology import MorphologyOperations
from components.geometryUtils import GeometryUtils


class MotionDetectionCustom:
    def __init__(
        self,
        min_area: int = 60,
        blur_kernel_size: int = 5,
        threshold: int = 25,
        bounding_threshold: int = 200,
        resize_coef: int = 4,
        max_frames: int = 1,
    ) -> None:
        """
        Initializes an instance of the MotionDetector class for motion detection on a video stream using custom realization.

        Parameters
        ----------
        min_area : int, optional
            Minimum area of an object to be considered as motion. Default is 500.

        blur_kernel_size : int, optional
            Size of the kernel for image blurring. Default is 5.

        threshold : int, optional
            Threshold value for motion detection (minimal difference for frames). Default is 25.

        bounding_threshold: int, optional
            Threshold value for dilated binary frame. Default is 200.

        resize_coef: int, optional
            A coefficient indicating how many times the image size will be compressed during image processing (used in dilatation). Default is 4.

        max_frames : int, optional
            Maximum number of frames stored for averaging past frames when defining object contours. Default is 4.
        """
        # detection parameters
        self.__min_area = min_area
        self.__blur_kernel_size = blur_kernel_size
        self.__threshold = threshold
        self.__max_frames = max_frames
        self.__resize_coef = resize_coef
        self.__bounding_threshold = bounding_threshold

        # detection components
        self.__gaussian_blur = GaussianBlur(self.__blur_kernel_size)
        self.__morphology = MorphologyOperations()

        # processed frame parameters
        self.__frame_count = 0
        self.__processed_frames = deque()

    def detect(
        self, old_gray_frame: np.ndarray, new_gray_frame: np.ndarray
    ) -> typing.List[typing.List[tuple]]:
        """
        Detects motion in a video stream based on the difference between two grayscale frames.

        Parameters
        ----------
        old_gray_frame : numpy.ndarray
            Grayscale representation of the previous frame.

        new_gray_frame : numpy.ndarray
            Grayscale representation of the current frame.

        Returns
        -------
        List[List[tuple]]
            A list of bounding rectangles [(x1, y1), (x2, y2)] representing the detected motion areas.
        """
        # when frame count is more than max computing frames, using in detect - update processed frames
        self.__frame_count += 1
        if self.__frame_count > self.__max_frames:
            self.__processed_frames.popleft()

        difference = np.abs(old_gray_frame.astype("int") - new_gray_frame).astype(
            "uint8"
        )
        difference = self.__gaussian_blur.blur_image(difference)
        threshold = ((difference > self.__threshold) * 255).astype("uint8")
        dilated = self.__morphology.dilate(threshold, self.__resize_coef)

        self.__processed_frames.append(dilated)

        accumulate_frame = np.mean(self.__processed_frames, axis=0).astype("uint8")
        contours = self.__find_contours(accumulate_frame)

        bounded_rectangles = []
        for contour in contours:
            left_up, right_down = contour
            left_up = (
                left_up[0] * self.__resize_coef,
                left_up[1] * self.__resize_coef,
            )
            right_down = (
                right_down[0] * self.__resize_coef,
                right_down[1] * self.__resize_coef,
            )
            bounded_rectangles.append([left_up, right_down])

        return bounded_rectangles

    def __find_contours(self, image: np.ndarray) -> typing.List[typing.List[tuple]]:
        """
        Find contours in a binary image using a contour finding algorithm based on breadth-first search.

        Parameters
        ----------
        image : np.ndarray
            Binary image where contours need to be identified.

        Returns
        -------
        List[List[int]]
            A list of contours, where each contour is represented by a list of
            coordinates [x, y].
        """
        h, w = image.shape[:2]
        visited = np.zeros((h, w), dtype=bool)
        in_image = lambda x, y: 0 <= x < w and 0 <= y < h
        contours = []
        queue = deque()

        # valid = np.where(image >= self.__bounding_threshold)
        # for row, col in zip(valid[0], valid[1]):

        for row in range(0, h, 2):
            for col in range(0, w, 2):
                if (
                    not visited[row][col]
                    and image[row][col]
                    and not GeometryUtils.dot_inside_contours(row, col, contours)
                ):
                    queue.append((col, row))
                    object_contour = [[col, row], [col, row]]

                    while queue:
                        x, y = queue.popleft()
                        visited[y, x] = True

                        neighbors = np.array(
                            [
                                (x - 1, y),  # left
                                (x, y - 1),  # up
                                (x + 1, y),  # right
                                (x, y + 1),  # down
                            ]
                        )

                        for neighbor in neighbors:
                            nx, ny = neighbor
                            if (
                                not visited[ny, nx]
                                and in_image(nx, ny)
                                and image[ny, nx] >= self.__bounding_threshold
                            ):
                                queue.append((nx, ny))
                                visited[ny, nx] = True

                                object_contour[0][0] = min(object_contour[0][0], nx)
                                object_contour[0][1] = min(object_contour[0][1], ny)
                                object_contour[1][0] = max(object_contour[1][0], nx)
                                object_contour[1][1] = max(object_contour[1][1], ny)

                    # check that area is above that min_area
                    if GeometryUtils.get_rect_area(object_contour) >= self.__min_area:
                        contours.append(object_contour)
        return contours
