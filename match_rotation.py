import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 2

parser = argparse.ArgumentParser(description='Template matcher')
parser.add_argument('--template', type=str, action='store',
                    help='The image to be used as template')
parser.add_argument('--map', type=str, action='store',
                    help='The image to be searched in')
parser.add_argument('--show', action='store_true',
                    help='Shows result image')
parser.add_argument('--save-dir', type=str, default='./',
                    help='Directory in which you desire to save the result image')

args = parser.parse_args()


def get_matched_coordinates(orig_image, template):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(orig_image, None)
    kp2, des2 = sift.detectAndCompute(template, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        w, h = orig_image.shape[::-1]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        template = cv2.polylines(
            template, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    img3 = cv2.drawMatches(orig_image, kp1, template, kp2,
                           good, None, **draw_params)

    # if --show argument used, then show result image
    # if args.show:
    plt.imshow(img3, 'gray'), plt.show()


orig_img_gray = cv2.imread("./images/parrot.png", 0)
# template_gray = cv2.imread("./images/parrot template.jpg", 0)
template_gray = cv2.imread("./images/rotated-parrot1.jpg", 0)

# equalize histograms
orig_img_eq = cv2.equalizeHist(orig_img_gray)
temp_img_eq = cv2.equalizeHist(template_gray)

# calculate matched coordinates
get_matched_coordinates(orig_img_eq, temp_img_eq)
