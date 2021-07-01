import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy


def process_a_channel(input, kernel):
    input = np.pad(input, ((1, 1), (1, 1)), 'constant')
    output = copy.deepcopy(input)
    h, w = input.shape
    for i in range(1, h - 1, 1):
        for j in range(1, w - 1, 1):
            roi = input[i - 1:i + 2, j - 1:j + 2]
            if kernel.sum() >= 9:
                output[i][j] = (roi * kernel).sum() // kernel.sum()
            else:
                output[i][j] = (roi * kernel).sum()

    return output[1:-1, 1:-1]


def visulize(img, kernel):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(img)

    for c in range(3):
        img[:, :, c] = process_a_channel(img[:, :, c], kernel)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(img)

    plt.show()


def visulize_sobel_cv2_api(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(img)

    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    plt.subplot(122)
    plt.axis('off')
    plt.imshow(dst)

    plt.show()

if __name__ == '__main__':
    img = cv2.imread('5.png')

    low_pass_mean = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])
    low_pass_weighted = np.array([[1, 1, 1],
                                  [1, 2, 1],
                                  [1, 1, 1]])
    low_pass_gaussian = np.array([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]])

    high_pass_laplacian = np.array([[0, -1, 0],
                                    [-1, 4, -1],
                                    [0, -1, 0]])
    high_pass_sobel_y = np.array([[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]])
    high_pass_prewiit_y = np.array([[-1, -1, -1],
                                  [0, 0, 0],
                                  [1, 1, 1]])
    high_pass_laplacian2 = np.array([[-1, -1, -1],
                                     [-1, 8, -1],
                                     [-1, -1, -1]])

    # visulize(img, low_pass_mean)

    visulize_sobel_cv2_api(img)
