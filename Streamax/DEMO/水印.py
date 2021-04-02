import cv2
import numpy as np
img1 = cv2.imread('../mydata/road.jpg')
overlay_img1 = np.ones(img1.shape, np.uint8) * 255

img2 = cv2.imread('../mydata/logo.jpg')
rows, cols, channels = img2.shape
overlay_img1[0:rows, 0:cols] = img2
img2gray = cv2.cvtColor(overlay_img1, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 55, cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)
temp1 = cv2.bitwise_and(img1, img1, mask=mask_inv)
temp2 = cv2.bitwise_and(overlay_img1, overlay_img1, mask=mask)
cv2.imshow('Temp1', temp1)
cv2.imshow('Temp2', temp2)
result = cv2.add(temp1, temp2)
cv2.imshow("Result", result)
cv2.imwrite("drawline/Result.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()