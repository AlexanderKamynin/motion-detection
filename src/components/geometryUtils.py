# import dependencies
import typing


class GeometryUtils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def rect_contain_another(
        rect1: typing.List[tuple], rect2: typing.List[tuple]
    ) -> bool:
        """
        Check if one rectangle is contained within another.

        Parameters
        ----------
        rect1 : List[tuple]
            Coordinates of the first rectangle in the format [(topLeft_x1, topLeft_y1), (downRight_x1, downRight_y1)].

        rect2 : List[tuple]
            Coordinates of the second rectangle in the format [(topLeft_x2, topLeft_y2), (downRight_x2, downRight_y2)].

        Returns
        -------
        bool
            True if rect1 contains rect2 or rect2 contains rect1, False otherwise.
        """
        (top_x1, top_y1), (down_x1, down_y1) = rect1
        (top_x2, top_y2), (down_x2, down_y2) = rect2

        if (
            top_x1 >= top_x2
            and down_x1 <= down_x2
            and top_y1 >= top_y2
            and down_y1 <= down_y2
        ):
            return True
        elif (
            top_x2 >= top_x1
            and down_x2 <= down_x1
            and top_y2 >= top_y1
            and down_y2 <= down_y1
        ):
            return True
        return False

    @staticmethod
    def get_rect_center(rect: typing.List[tuple]) -> tuple:
        """
        Get the center coordinates of a rectangle.

        Parameters
        ----------
        rect : List[tuple]
            Coordinates of the rectangle in the format [(topLeft_x, topLeft_y), (downRight_x, downRight_y)].

        Returns
        -------
        tuple
            The coordinates of the center of the rectangle in the format (center_x, center_y).
        """
        x = (rect[1][0] + rect[0][0]) // 2
        y = (rect[1][1] + rect[0][1]) // 2
        return (x, y)

    @staticmethod
    def dot_inside_contours(
        x: int, y: int, contours: typing.List[typing.List[tuple]]
    ) -> bool:
        """
        Check if a point is inside any of the contours.

        Parameters
        ----------
        x : int
            X-coordinate of the point.

        y : int
            Y-coordinate of the point.

        contours : List[List[tuple]]
            List of contours, where each contour is represented as [(topLeft_x, topLeft_y), (downRight_x, downRight_y)].

        Returns
        -------
        bool
            True if the point is inside any of the contours, False otherwise.
        """
        for rect in contours:
            if rect[0][0] <= x <= rect[1][0] and rect[0][1] <= y <= rect[1][1]:
                return True
        return False

    @staticmethod
    def dot_inside_contour(x: int, y: int, contour: typing.List[tuple]) -> bool:
        if contour[0][0] <= x <= contour[1][0] and contour[0][1] <= y <= contour[1][1]:
            return True
        return False

    @staticmethod
    def get_rect_area(rect: typing.List[tuple]) -> int:
        """
        Calculate the area of the specified rectangle.

        Parameters
        ----------
        rect : List[tuple]
            Coordinates of the rectangle in the format [(topLeft_x, topLeft_y), (downRight_x, downRight_y)].

        Returns
        -------
        int
            The area of the rectangle.
        """
        return (rect[1][0] - rect[0][0]) * (rect[1][1] - rect[0][1])
