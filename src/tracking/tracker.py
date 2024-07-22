# import dependencies
import cv2
import numpy as np
import typing
import sys
import os

# import project components
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.components.imageUtils import ImageProcessingUtils


class Tracker:
    def __init__(
        self,
        frame_size: typing.Tuple[int, int, int],
        max_points: int = 30,
        win_size: typing.Tuple[int, int] = (15, 15),
    ) -> None:
        """
        Initialize an instance of the Tracker class for optical flow tracking using the Lucas-Kanade method.

        Parameters
        ----------
        frame_size : Tuple[int, int, int]
            The size of the video frames (height, width, num_channels).

        max_points : int, optional
            The count of max tracking points to save. Default is 30.

        win_size : Tuple[int, int], optional
            The size of the window used to find the optical flow. Default is (15,15).
        """
        self.__win_size = win_size

        """
            Parameters to execute the Lucas-Kanade method
            1. winSize: The size of the window used to find the optical flow.
                This window is shifted from one frame to another.
            2. maxLevel: The maximum level of the pyramid used to find the optical flow.
                This allows for multiple levels of optical flow alignment to improve accuracy.
            3. criteria: A stopping criterion for the iterative optical flow algorithm. 
                This is usually a combination of the maximum number of iterations and the minimum value used to estimate the change.
        """
        self.__lk_params = dict(
            winSize=self.__win_size,
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.__frame_size = frame_size
        self.__thickness = 3
        self.__mask = np.zeros(self.__frame_size)
        self.__max_point = max_points
        self.__points_history = {}

    def track(
        self,
        old_gray_frame: np.ndarray,
        new_gray_frame: np.ndarray,
        object_points: typing.Dict[int, typing.Tuple[int, int]],
    ) -> np.ndarray:
        """
        Perform optical flow tracking using the Lucas-Kanade method.

        Parameters
        ----------
        old_gray_frame : numpy.ndarray
            Grayscale previous frame.

        new_gray_frame : numpy.ndarray
            Grayscale current frame.

        object_points : Dict[int, Tuple[int, int]]
            Dictionary containing the unique identifiers of the objects as keys and their corresponding rectangle center coords as values in the form (x, y).

        Returns
        -------
        numpy.ndarray
            Mask (image) with vectors from old to new positions.
        """
        if not object_points:
            return self.__mask.astype("uint8")

        old_points = np.array(list(object_points.values()), dtype=np.float32)
        new_points, _, _ = cv2.calcOpticalFlowPyrLK(
            old_gray_frame, new_gray_frame, old_points, None, **self.__lk_params
        )

        # save points
        for idx, id in enumerate(object_points.keys()):
            if not id in self.__points_history.keys():
                self.__points_history[id] = []
            self.__points_history[id].append(
                {
                    "old_points": old_points[idx],
                    "new_points": new_points[idx],
                }
            )

        for idx, id in enumerate(object_points.keys()):
            old = tuple(map(int, old_points[idx]))
            new = tuple(map(int, new_points[idx]))
            self.__mask = cv2.line(
                self.__mask,
                old,
                new,
                ImageProcessingUtils.generate_random_color(id),
                thickness=self.__thickness,
            )

        self.__clear_last_points(set(object_points.keys()))

        return self.__mask.astype("uint8")

    def optical_flow(
        self,
        old_frame: np.ndarray,
        new_frame: np.ndarray,
        object_points: typing.List[typing.List[tuple]],
    ):
        """
        The method is not used anywhere at the moment
        """
        w = self.__win_size[0] // 2

        # normalize images
        old_frame = old_frame / 255.0
        new_frame = new_frame / 255.0

        for point in object_points:
            x, y = point

            Ix = 0
            Iy = 0
            It = 0
            for k in range(-w, w + 1):
                Ix += old_frame[y, x + k + w] - old_frame[y, x + k - w]
                Iy += old_frame[y + k + w, x] - old_frame[y + k - w, x]
                It += new_frame[y + k, x + k] - old_frame[y + k, x + k]

            b = np.array([-It])
            A = np.array([[Ix, Iy]])

            U = np.matmul(np.linalg.pinv(A), b)

            print(f"--custom: old: x={x}, y={y}; new: x={x + U[0]}, y={y + U[1]}")

    def __clear_last_points(self, detected_ids: set) -> None:
        """
        Clear the last tracked points from the history.

        Returns
        -------
        None
        """

        # define id that already not on the image
        not_used_id = set(self.__points_history.keys()) - detected_ids

        empty_id = []
        for id in self.__points_history.keys():
            point_count = len(self.__points_history[id])
            if point_count >= self.__max_point or id in not_used_id:
                last_points = self.__points_history[id].pop(0)

                if point_count - 1 == 0:
                    empty_id.append(id)

                old = last_points["old_points"]
                new = last_points["new_points"]
                # set black on the mask
                cv2.line(
                    self.__mask,
                    tuple(map(int, old)),
                    tuple(map(int, new)),
                    color=(0, 0, 0),
                    thickness=self.__thickness,
                )

        # delete all empty points id
        for id in empty_id:
            del self.__points_history[id]
