import cv2


def match(img, temp):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    w, h = temp.shape[::]
    res = cv2.matchTemplate(grey_img, temp, cv2.TM_SQDIFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + h, top_left[1] + w)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    cv2.imshow("match", img)
    cv2.waitKey()

    cv2.destroyAllWindows()


match(cv2.imread("./images/parrot.png"), cv2.imread("./images/parrot template.jpg", cv2.IMREAD_GRAYSCALE))


# import numpy as np
# import imutils
# import cv2
#
#
# template = cv2.imread("./images/parrot template.jpg")
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = cv2.Canny(template, 50, 200)
# (tH, tW) = template.shape[:2]
# cv2.imshow("Template", template)
#
# image = cv2.imread("./images/parrot.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# found = None
# for scale in np.linspace(0.2, 1.0, 20)[::-1]:
#     resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
#     r = gray.shape[1] / float(resized.shape[1])
#     if resized.shape[0] < tH or resized.shape[1] < tW:
#         break
#     edged = cv2.Canny(resized, 50, 200)
#     result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
#     (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
#     clone = np.dstack([edged, edged, edged])
#     new_scale = (int(maxLoc[0] + (tW / scale)), int(maxLoc[1] + (tH / scale)))
#     cv2.rectangle(clone, maxLoc,
#                   new_scale, (0, 0, 255), 2)
#     # cv2.imshow("Visualize", clone)
#     cv2.waitKey(0)
#     if found is None or maxVal > found[0]:
#         found = (maxVal, maxLoc, r)
# (_, maxLoc, r) = found
# (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
# (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
# cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
