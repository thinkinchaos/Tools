import os
import cv2
import numpy as np
import colour_demosaicing
from matplotlib import pyplot as plt
import math


def mkdir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def myISP(input, raw_shape=(3456, 4608), awg_gain=(1, 1, 1), ccm=([1.712891, -0.714844, 0.001953],
                                                                  [-0.164062, 1.265625, -0.101562],
                                                                  [0.126953, -0.814453, 1.687500]), gamma=2.2):
    if isinstance(input, str):
        raw = np.fromfile(input, dtype='float32')
        bayer_raw = raw.reshape(raw_shape)
        bayer_raw = np.transpose(bayer_raw)
    else:
        bayer_raw = input

    rgb = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(bayer_raw, 'BGGR')
    rgb = np.clip(rgb, 0, 1)

    Iawb = rgb
    for i in range(3):
        Iawb[:, :, i] = rgb[:, :, i] * awg_gain[i]
    Iawb = np.clip(Iawb, 0, 1)

    Iccm = Iawb
    for i in range(3):
        Iccm[:, :, i] = Iawb[:, :, 0] * ccm[i][0] + Iawb[:, :, 1] * ccm[i][1] + Iawb[:, :, 2] * ccm[i][2]
    Iccm = np.clip(Iccm, 0, 1)

    Igamma = np.power(Iccm, 1.0 / gamma)

    Iout = (Igamma * 255).astype('uint8')
    return Iout


def get_edge_by_pyramid_raw(raw_path, raw_shape=(4344, 5792), black_level=64, white_level=1023, eps=0.000001):
    raw = np.fromfile(raw_path, dtype='float32')
    imgraw = raw.reshape(raw_shape)
    imgraw = np.transpose(imgraw)
    diff = white_level - black_level
    imgraw = np.maximum(imgraw, 0)

    Iin = myISP(imgraw)

    imgraw = imgraw * diff  # 为什么乘白×黑？
    b = imgraw[0::2, 0::2]
    gb = imgraw[0::2, 1::2]
    gr = imgraw[1::2, 0::2]
    r = imgraw[1::2, 1::2]
    g = (gb + gr) / 2

    rgb = np.zeros((b.shape[0], b.shape[1], 3), b.dtype)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    print('Data Range', min(rgb.flatten()), max(rgb.flatten()))

    Y = 0.299 * r + 0.587 * g + 0.114 * b
    U = -0.169 * r - 0.331 * g + 0.500 * b
    V = 0.500 * r - 0.419 * g - 0.081 * b
    I_lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    print('Mean of YUVLum', np.mean(Y), np.mean(U), np.mean(V), np.mean(I_lum))

    input_name = 'Y'

    if input_name == 'Y':
        save_dir = 'Y'
        input = Y
    elif input_name == 'U':
        save_dir = 'U'
        input = U
    elif input_name == 'V':
        save_dir = 'V'
        input = V
    else:
        assert input_name == 'I_lum'
        save_dir = 'I_lum'
        input = I_lum
    mkdir(save_dir)
    cv2.imwrite(save_dir + '/' + save_dir + '_BeforeDenoise.png', cv2.cvtColor(Iin, cv2.COLOR_RGB2BGR))

    I_filt = cv2.GaussianBlur(input, (5, 5), 1.5)
    I_log = np.log(I_filt + eps)  # 为什么高斯模糊和对数变换，为了求梯度更有用吗？

    # print(I_log.shape)

    def multiscale_gradient_cv2(input, lowPassKS, alphaFact, beta, layers):
        gX = np.array(([0, 0, 0, ], [-1, 0, 1], [0, 0, 0]))
        gY = np.array(([0, -1, 0, ], [0, 0, 0], [0, 1, 0]))
        pyramid_maps = {'X': [], 'Y': []}
        Lt = input
        for i in range(1, layers + 1):
            Gx = cv2.filter2D(Lt, -1, gX) / np.power(2, i)
            Gy = cv2.filter2D(Lt, -1, gY) / np.power(2, i)
            pyramid_maps['X'].append(Gx)
            pyramid_maps['Y'].append(Gy)
            if i < layers:
                Lt = cv2.blur(Lt, (lowPassKS, lowPassKS))  # 下采样之前低通滤波，也为了求梯度吗？为什么不用高斯了？
                Lt = cv2.resize(Lt, None, fx=0.5, fy=0.5)
        sqG2 = np.sqrt(pyramid_maps['X'][0] ** 2 + pyramid_maps['Y'][0] ** 2)
        avGr = np.mean(sqG2)
        alpha = avGr * alphaFact  # 平均梯度乘以一个系数，

        # print(alpha)# 0.01

        def attenuation_mask(Gx, Gy, alpha, beta):
            gradNorm = np.sqrt(Gx ** 2 + Gy ** 2)
            # print(np.mean(gradNorm)) #0.01~0.05，图像越小，梯度越大
            a = alpha / (gradNorm + eps)  # 系数/梯度图，
            b = gradNorm / alpha  # 梯度图/系数，放大一百倍
            mask = a * (b ** beta)  # 约为(1/g)的0.15次方，梯度越大，掩码越小.梯度越小的地方值越大，相当于求平坦区的掩码？
            # print(np.mean(mask)) #1.2~0.9
            # mask[gradNorm == 0] = 1  # 梯度为0的平坦区直接赋值为1，将掩码值封顶
            return mask

        phiKp1 = attenuation_mask(pyramid_maps['X'][layers - 1], pyramid_maps['Y'][layers - 1], alpha,
                                  beta)  # 对最小层求衰减掩码
        # print('init', phiKp1.shape)
        cv2.imwrite(save_dir + '/' + save_dir + '_0_layer.jpg', phiKp1 * 255)

        for i in range(1, layers):
            j = layers - i - 1  # 从次高层开始，直到未下采样层
            h, w = pyramid_maps['X'][j].shape
            # print(j, h,w)
            phiK = attenuation_mask(pyramid_maps['X'][j], pyramid_maps['Y'][j], alpha, beta)  # 对该层求稍大一层的平坦区
            cv2.imwrite(save_dir + '/' + save_dir + '_' + str(j + 1) + '_layer.jpg', phiK * 255)
            # print('a',phiKp1.shape)
            phiKp1 = cv2.resize(phiKp1, (w, h)) * phiK  # 将最小层放大到该层，并乘以该层的掩码，最为新的最小层
            # print('b',phiKp1.shape)
        return phiKp1

    gradient_mask = multiscale_gradient_cv2(I_log, lowPassKS=5, alphaFact=0.2, beta=0.85, layers=5)  # 最终平坦区掩码为原图大小
    # print('final', gradient_mask.shape)
    gradient_mask = gradient_mask ** 2  # 使小分数更小，越平坦的地方越重视

    img_mask = 1 - gradient_mask  # 取反，变为纹理区的掩码，纹理越强的地方越重视
    img_mask = np.minimum(1, np.maximum(img_mask, 0))

    # base = 0.2
    # t = 2
    # img_mask_n = np.maximum(img_mask - base, 0)  # 为什么减去base
    # img_mask_n = np.power(img_mask_n, t)  # 强纹理的贡献继续增大
    # smooth_mask = 1 - img_mask_n  # 取反，获得强纹理的掩码？
    # smooth_mask = np.maximum(smooth_mask - 0.3, 0) ** 2  # 贡献增强，但为什么减去0.3
    # cv2.imwrite('out1.jpg', smooth_mask * 255)
    # cv2.imwrite('out2.jpg', img_mask_n * 255)

    imgraw[0::2, 0::2] = cv2.GaussianBlur(b, (5, 5), 1) * img_mask + b * (1 - img_mask)  # 对强纹理区域去噪，其他区域保护纹理不做处理
    imgraw[0::2, 1::2] = cv2.GaussianBlur(gb, (5, 5), 1) * img_mask + gb * (1 - img_mask)
    imgraw[1::2, 0::2] = cv2.GaussianBlur(gr, (5, 5), 1) * img_mask + gr * (1 - img_mask)
    imgraw[1::2, 1::2] = cv2.GaussianBlur(r, (5, 5), 1) * img_mask + r * (1 - img_mask)
    imgraw /= diff

    Iout = myISP(imgraw)
    cv2.imwrite(save_dir + '/' + save_dir + '_AfterDenoise.png', cv2.cvtColor(Iout, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    # raw_path = 'supernight_pipeline_fusion_out_bayer_4608_3456_1_float.raw'
    # raw_shape =
    # awg_gain = [1, 1, 1]
    # ccm = [[1.712891, -0.714844, 0.001953],
    #        [-0.164062, 1.265625, -0.101562],
    #        [0.126953, -0.814453, 1.687500]]
    # gamma = 2.2
    # Iout = myISP(raw_path, raw_shape, awg_gain, ccm, gamma)
    # Iout =
    # cv2.namedWindow('Iout', 0)
    # cv2.imshow('Iout', Iout)
    # cv2.waitKey(0)

    get_edge_by_pyramid_raw('IMG_20200706_213100.raw')
