import cv2
import numpy as np


def match(img, temp, threshold):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    w, h = temp.shape[::-1]
    res = cv2.matchTemplate(grey_img, temp, cv2.TM_CCOEFF_NORMED)

    # Find the brightest pixel to start looking for the match
    loc = np.where(res >= threshold)
    points = zip(*loc[::-1])
    for i in points:
        print(i)
        # draw the rectangle starting from this pixel
        cv2.rectangle(img, i, (i[0] + w, i[1] + h), (0, 0, 255), 1)

    cv2.imshow("img", img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


# match(cv2.imread("./images/messi.jpg"), cv2.imread("./images/messi-face.jpg", cv2.IMREAD_GRAYSCALE), 0.61) match(
# cv2.imread("./images/color_pencils.jpeg"), cv2.imread("./images/color-pencils-template.jpg", cv2.IMREAD_GRAYSCALE),
# 0.68) match(cv2.imread("./images/flower-field.jpeg"), cv2.imread("./images/flower-field-template2.jpg",
# cv2.IMREAD_GRAYSCALE), 0.5)
match(cv2.imread("./images/flower-field.jpeg"), cv2.imread("./images/flower-field-template.jpg", cv2.IMREAD_GRAYSCALE), 0.797)



