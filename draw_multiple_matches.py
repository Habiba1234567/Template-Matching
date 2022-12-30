import cv2
import numpy as np


def draw_multiple_matches(img, temp, threshold):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    w, h = temp.shape[::-1]

    res = cv2.matchTemplate(grey_img, temp, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    points = zip(*loc[::-1])
    for i in points:
        cv2.rectangle(img, i, (i[0] + w, i[1] + h), (0, 255, 255), 1)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


draw_multiple_matches(cv2.imread('./images/card.jpeg'), cv2.imread("./images/card-template.jpg", 0), 0.7)
