import cv2 as cv
import numpy as np
import src.utils.ColorUtils as clr


def crop_image_by_contour(image, contour):
    mask = __create_crop_mask(image, contour)
    crop_coordinate = __get_crop_coordinate(contour)
    crop_image = np.zeros((crop_coordinate.get_height(), crop_coordinate.get_width(), 3), dtype="uint8")
    for i in range(crop_coordinate.get_top(), crop_coordinate.get_bottom()):
        for j in range(crop_coordinate.get_left(), crop_coordinate.get_right()):
            pixel_color = mask[i][j]
            y = i - crop_coordinate.get_top()
            x = j - crop_coordinate.get_left()
            if clr.is_white(pixel_color):
                crop_image[y][x] = image[i][j]
    return crop_image


def __create_crop_mask(image, contour):
    mask = np.zeros(image.shape)
    cv.drawContours(mask, [contour], -1, clr.get_white_color(), -1)
    return mask


def __get_crop_coordinate(contour):
    top, bottom, left, right = 0, 0, 0, 0
    for point in contour:
        y = point[0][1]
        x = point[0][0]
        if bottom < y:
            bottom = y
        if top > y or top == 0:
            top = y
        if right < x:
            right = x
        if left > x or left == 0:
            left = x
    return CropCoordinate(top, bottom, left, right)


class CropCoordinate:

    def __init__(self, top, bottom, left, right):
        self._top = top
        self._bottom = bottom
        self._left = left
        self._right = right

    def get_top(self):
        return self._top

    def get_bottom(self):
        return self._bottom

    def get_left(self):
        return self._left

    def get_right(self):
        return self._right

    def get_height(self):
        return self._bottom - self._top

    def get_width(self):
        return self._right - self._left
