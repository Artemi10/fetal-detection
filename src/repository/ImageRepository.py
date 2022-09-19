import cv2 as cv


def read_image(path):
    return cv.imread("../res/image/"+path)


def write_image(image, path):
    cv.imwrite("../res/detected/"+path, image)
