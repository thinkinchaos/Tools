import cv2
import numpy as np
from scipy import interpolate


def draw_curve_fig(img, y, ax_length=100, margin=10):
    p_00 = (margin, ax_length + margin)
    p_01 = (margin, margin)
    p_10 = (margin + ax_length, ax_length + margin)
    cv2.line(img, p_00, p_01, (255, 0, 0), thickness=2)
    cv2.line(img, p_00, p_10, (255, 0, 0), thickness=2)
    p_01_l = (p_01[0] - 1, p_01[1] + 3)
    p_01_r = (p_01[0] + 1, p_01[1] + 3)
    p_10_u = (p_10[0] - 3, p_10[1] - 1)
    p_10_d = (p_10[0] - 3, p_10[1] + 1)
    cv2.line(img, p_01, p_01_l, (255, 0, 0), thickness=2)
    cv2.line(img, p_01, p_01_r, (255, 0, 0), thickness=2)
    cv2.line(img, p_10, p_10_u, (255, 0, 0), thickness=2)
    cv2.line(img, p_10, p_10_d, (255, 0, 0), thickness=2)
    cv2.putText(img, 'fps', (p_10[0] + 10, p_10[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    x = []
    last_num = y[-1]
    for i in range(len(y)):
        x.append(margin + (ax_length // len(y)) * i)
    y = np.array(y)
    y = ax_length - ((y / max(y))) * ax_length + margin
    y = [int(i) for i in y]
    tck = interpolate.splrep(x, y)
    xx = np.linspace(min(x), max(x), 100).astype(int)
    yy = interpolate.splev(xx, tck, der=0).astype(int)
    for i in range(len(xx) - 1):
        p_cur = (xx[i] + 3, yy[i])
        p_next = (xx[i + 1] + 3, yy[i + 1])
        cv2.line(img, p_cur, p_next, (125, 255, 0), thickness=1)
    cv2.putText(img, str(last_num), (xx[-1] + 10, yy[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img


img = cv2.imread('D:/1.jpg')
nums = [56, 12, 23, 77, 99, 43, 21, 90, 30, 45]
img = draw_curve_fig(img, nums)
cv2.imshow('s', img)
cv2.waitKey()
