import cv2 as cv


def detect_ultrasound_image_contour(image):
    gray_image = __generate_gray_image(image)
    mask = __generate_threshold_image(gray_image)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return max(contours, key=cv.contourArea)


def __generate_gray_image(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def __generate_threshold_image(image):
    ret, thresh = cv.threshold(image, 0.0, 255.0, cv.THRESH_TRIANGLE)
    return thresh
