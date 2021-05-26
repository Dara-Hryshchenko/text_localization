import os

import cv2
import numpy as np
INPUT_DIRECTORY_PATH = 'input'
OUTPUT_DIRECTORY_PATH = 'output'
# how many times look for smaller regions with text inside big contours
MAX_N_ITER = 1


def localize_text(img_path):
    img = cv2.imread(img_path)
    rgb_small = cv2.pyrDown(img)
    return draw_text_box(rgb_small)


def draw_text_box(img, n_iter=0):
    small = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        if r > 0.2 and w > 40 and h > 40 and n_iter < MAX_N_ITER:
            # find text inside huge contour
            img_slice = draw_text_box(img[y + 5:y + h - 5, x + 5:x + w - 5], n_iter=n_iter + 1)
            img[y+5:y+h-5, x+5:x+w-5] = img_slice
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)

        elif w > 10 and h > 10 and r > 0.35:
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)

    return img


if __name__ == "__main__":
    images = os.listdir(INPUT_DIRECTORY_PATH)
    for img_name in images:
        result_img = localize_text(os.path.join(INPUT_DIRECTORY_PATH, img_name))
        cv2.imshow(img_name, cv2.resize(result_img, (640, 960)))
        cv2.imwrite(os.path.join(OUTPUT_DIRECTORY_PATH, img_name), result_img)
    cv2.waitKey()
