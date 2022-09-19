import torch
import numpy as np
from PIL import Image
import cv2 as cv
import src.utils.ColorUtils as clr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Loads to the GPU


def detect_fetal_image(ultrasound_image):
    model = torch.load('../res/model/model.pt')
    data = _create_evaluation_data(ultrasound_image)
    predictions = model(data)
    mask_array = predictions.reshape(572, 572).detach().cpu().numpy()
    mask = cv.cvtColor(np.array(Image.fromarray(np.uint8(mask_array * 255))), cv.COLOR_RGB2BGR)
    img_array = data.reshape(572, 572).detach().cpu().numpy()
    img = cv.cvtColor(np.array(Image.fromarray(np.uint8(img_array))), cv.COLOR_RGB2BGR)
    return _add_fetal_contour(img, mask)


def _add_fetal_contour(image, mask):
    contours, hierarchy = cv.findContours(cv.cvtColor(mask, cv.COLOR_BGR2GRAY), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    _draw_ellips(image, sorted_contours[0])
    if len(sorted_contours[0]) * 0.4 <= len(sorted_contours[1]):
        _draw_ellips(image, sorted_contours[1])
    return image


def _draw_ellips(image, contour):
    cv.drawContours(image, [contour], -1, clr.get_red_color(), 3)


def _create_evaluation_data(ultrasound_image):
    gray_image = cv.cvtColor(ultrasound_image, cv.COLOR_BGR2GRAY, 1)
    resized_image = cv.resize(gray_image, (572, 572), interpolation=cv.INTER_LINEAR)
    data = np.zeros((1, 1,) + resized_image.shape)
    data[0][0] = resized_image
    return torch.from_numpy(data).float().to(device)
