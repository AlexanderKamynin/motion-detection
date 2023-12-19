import typing

class Geometry:
    def __init__(self):
        pass
    
    @staticmethod 
    def rect_contain_another(rect1: typing.List[tuple], rect2: typing.List[tuple]):
        (top_x1, top_y1), (down_x1, down_y1) = rect1
        (top_x2, top_y2), (down_x2, down_y2) = rect2

        if top_x1 >= top_x2 and down_x1 <= down_x2 and top_y1 >= top_y2 and down_y1 <= down_y2:
            return True
        elif top_x2 >= top_x1 and down_x2 <= down_x1 and top_y2 >= top_y1 and down_y2 <= down_y1:
            return True
        return False
    
    @staticmethod
    def get_rect_center(rect: typing.List[tuple]):
        x = (rect[1][0] + rect[0][0]) // 2
        y = (rect[1][1] + rect[0][1]) // 2
        return (x, y)
