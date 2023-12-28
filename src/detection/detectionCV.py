# import dependencies
import cv2
import numpy as np
from collections import deque
import typing
import os
import sys

# import project components
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from components.geometryUtils import GeometryUtils


class MotionDetectionCV:
    def __init__(
        self,
        min_area: int = 500,
        blur_kernel_size: int = 5,
        threshold: int = 25,
        max_frames: int = 4,
    ):
        """
        Initializes an instance of the MotionDetector class for motion detection on a video stream using the OpenCV library.

        Parameters
        ----------
        min_area : int, optional
            Minimum area of an object to be considered as motion. Default is 500.

        blur_kernel_size : int, optional
            Size of the kernel for image blurring. Default is 5.

        threshold : int, optional
            Threshold value for motion detection. Default is 25.

        max_frames : int, optional
            Maximum number of frames stored for averaging past frames when defining object contours. Default is 4.
        """
        # detection parameters
        self.__min_area = min_area
        self.__blur_kernel_size = (blur_kernel_size, blur_kernel_size)
        self.__threshold = threshold
        self.__max_frames = max_frames

        # processed frame parameters
        self.__frame_count = 0
        self.__processed_frames = deque()
        self.__detected_objects = {}

    def detect(
        self, old_gray_frame: np.ndarray, new_gray_frame: np.ndarray
    ) -> typing.Dict[int, typing.List[tuple]]:
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
        Dict[int, List[tuple]]
            A dictionary containing the detected motion areas represented as a list of bounding rectangles [(x1, y1), (x2, y2)].
            The keys of the dictionary correspond to the unique identifications of the detected objects.
        """
        # when frame count is more than max computing frames, using in detect - update processed frames
        self.__frame_count += 1
        if self.__frame_count > self.__max_frames:
            self.__processed_frames.popleft()

        difference = cv2.absdiff(old_gray_frame, new_gray_frame)
        difference = cv2.GaussianBlur(difference, self.__blur_kernel_size, 0)
        _, threshold = cv2.threshold(
            difference, self.__threshold, 255, cv2.THRESH_BINARY
        )
        dilated = cv2.dilate(threshold, None, iterations=3)
        self.__processed_frames.append(dilated)

        accumulate_frame = np.mean(self.__processed_frames, axis=0).astype("uint8")
        # cv2.RETR_EXTERNAL provide deleting all inner (daughter) contours
        contours, _ = cv2.findContours(
            accumulate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # contouring area around the detected objects
        bounded_rectangles = []
        for contour in contours:
            if cv2.contourArea(contour) < self.__min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            bounded_rectangles.append([(x, y), (x + w, y + h)])

        bounded_rectangles = self.__delete_inner_rectangles(bounded_rectangles)
        self.__update_objects_id(bounded_rectangles)

        return self.__detected_objects

    def __update_objects_id(
        self, bounded_rectangles: typing.List[typing.List[tuple]]
    ) -> None:
        """
        Update the dictionary of detected objects based on a list of bounded rectangles.

        Parameters
        ----------
        bounded_rectangles : List[List[Tuple]]
            A list of bounded rectangles, where each rectangle is represented as a list of tuples

        Returns
        -------
        None
        """
        object_ids = self.__detected_objects.keys()
        unused_detected_id = set(object_ids)

        if not self.__detected_objects:
            for idx, rect in enumerate(bounded_rectangles):
                self.__detected_objects[idx] = rect.copy()
        else:
            for idx, rect in enumerate(bounded_rectangles):
                match_found = False

                for id in object_ids:
                    object_center = GeometryUtils.get_rect_center(
                        self.__detected_objects[id]
                    )

                    if GeometryUtils.dot_inside_contour(
                        x=object_center[0], y=object_center[1], contour=rect
                    ):
                        self.__detected_objects[id] = rect
                        # mark that rect object not new
                        match_found = True
                        # mark that id is actual
                        unused_detected_id.remove(id)
                        break

                # add the new if no match found
                if not match_found:
                    new_id = max(object_ids) + 1
                    self.__detected_objects[new_id] = rect.copy()

        # remove objects that already no on the frame
        for id in unused_detected_id:
            del self.__detected_objects[id]

    def __delete_inner_rectangles(
        self, bounded_rectangles: typing.List[typing.List[tuple]]
    ) -> typing.List[typing.List[tuple]]:
        """
        Delete inner rectangles from a list of bounding rectangles.

        Parameters
        ----------
        bounded_rectangles : List[List[Tuple[int, int]]]
        A list of bounding rectangles [(x1, y1), (x2, y2)].

        Returns
        -------
        List[List[Tuple[int, int]]]
        A filtered list of bounding rectangles without inner rectangles.
        """
        if not len(bounded_rectangles):
            return

        # it's more likely that smaller rectangles will fall into the other ones
        rect_count = len(bounded_rectangles)
        bounded_rectangles = sorted(
            bounded_rectangles,
            key=lambda rect: (rect[1][0] - rect[0][0]) * (rect[1][1] - rect[0][1]),
            reverse=True,
        )  # sort by rectangles area

        is_inner = [True] * rect_count
        for i in range(rect_count):
            if is_inner[i]:
                for j in range(i + 1, rect_count):
                    if is_inner[j] and GeometryUtils.rect_contain_another(
                        bounded_rectangles[i], bounded_rectangles[j]
                    ):
                        is_inner[j] = False

        bounded_rectangles = [
            bounded_rectangles[i] for i in range(rect_count) if is_inner[i]
        ]

        return bounded_rectangles
