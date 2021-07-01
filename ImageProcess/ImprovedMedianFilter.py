from matplotlib import pyplot as plt
import numpy as np
import cv2
import copy
import math


def random_noise(img_noise, noise_num):
    rows, cols, chn = img_noise.shape
    # 加噪声
    for i in range(noise_num):
        x = np.random.randint(0, rows)  # 随机生成指定范围的整数
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise


def gasuss_noise(image, mean=0, var=0.01):
    image = np.array(image / 255, dtype=float)  # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    noise = np.random.normal(mean, var ** 0.5, image.shape)  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    out = image + noise  # 将噪声和原始图像进行相加得到加噪后的图像
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)  # clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
    out = np.uint8(out * 255)  # 解除归一化，乘以255将加噪后的图像的像素值恢复
    return out


def median_blur_ch1(img, kernel_size, mode):
    height, width = img.shape[:2]
    offset = kernel_size // 2
    img_padded = np.lib.pad(img, offset, mode='symmetric')
    img_out = np.zeros((height, width), dtype=img.dtype)
    for x in range(height):
        for y in range(width):
            cell = img_padded[x:x + kernel_size, y:y + kernel_size]
            # print(cell.shape)

            if mode == 'four':
                vector_h = cell[offset, :].flatten()
                vector_v = cell[:, offset].flatten()
                vector_l, vector_r = [], []
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        if i == j:
                            vector_l.append(cell[i][j])
                        if i + j == kernel_size - 1:
                            vector_r.append(cell[i][j])
                # print(len(vector_l), len(vector_r), len(vector_v), len(vector_h))

                var_dict = dict()
                var_dict.setdefault('h', np.var(vector_h))
                var_dict.setdefault('v', np.var(vector_v))
                var_dict.setdefault('l', np.var(vector_l))
                var_dict.setdefault('r', np.var(vector_r))

                vars = [np.var(vector_h), np.var(vector_v), np.var(vector_l), np.var(vector_r)]
                vectors = [vector_h, vector_v, vector_l, vector_r]
                min_var = np.min(vars)
                min_var_index = vars.index(min_var)
                min_vector = vectors[min_var_index]
                # print(max_vector)

                min_vector.sort()
                img_out[x, y] = min_vector[len(min_vector) // 2]
            else:
                vector = cell.flatten()
                vector.sort()
                img_out[x, y] = vector[(kernel_size * kernel_size) // 2]
    return img_out


def judge_low_frequency(block, thresh, method='sobel'):
    assert len(block.shape) == 2
    block = cv2.GaussianBlur(block, (5, 5), 0)
    low_frequency_flag = True
    if method == 'sobel':
        grad_X = cv2.Sobel(block, -1, 1, 0)
        grad_Y = cv2.Sobel(block, -1, 0, 1)
        grad = cv2.addWeighted(grad_X, 0.5, grad_Y, 0.5, 0)
        avg_grad = np.mean(grad.flatten())
        # cv2.imshow('ss', grad)
        # cv2.waitKey()
        # print(avg_grad)
        if avg_grad > thresh:
            low_frequency_flag = False
    else:
        block_dct = cv2.dct(np.float32(block))
        block_dct[block_dct < 0] = 0
        block_dct_mean = np.mean(block_dct.flatten())
        print(block_dct_mean)
        if block_dct_mean < thresh:
            low_frequency_flag = False
    return low_frequency_flag


def median_based_denoise_ch1(noised_img, thresh, block_div, kernel_size):
    assert len(noised_img.shape) == 2
    h, w = noised_img.shape
    block_h = h // block_div
    block_w = w // block_div
    offset = kernel_size // 2

    for y in range(0, h, block_h):
        for x in range(0, w, block_w):
            block = noised_img[y:y + block_h, x:x + block_w]
            if block.shape[0] != block_h or block.shape[1] != block_w:
                continue

            low_frequency_flag = judge_low_frequency(block, thresh)

            if low_frequency_flag:
                # if y - offset >= 0 and x - offset >= 0 and y + block_h + offset <= h and x + block_w + offset <= w:
                #     blur_block = noised_img[y - offset:y + block_h + offset, x - offset:x + block_w + offset]
                #     tmp = median_blur_ch1(blur_block, kernel_size, mode='default')
                #     noised_img[y:y + block_h, x:x + block_w] = tmp[offset:-offset, offset:-offset]
                # else:
                #     blur_block = noised_img[y:y + block_h, x:x + block_w]
                #     noised_img[y:y + block_h, x:x + block_w] = median_blur_ch1(blur_block, kernel_size, mode='default')
                pass

                # cv2.rectangle(noised_img, (x, y), (x + block_w, y + block_h), (0, 0, 255), 2)
            else:
                # if y - offset >= 0 and x - offset >= 0 and y + block_h + offset <= h and x + block_w + offset <= w:
                #     blur_block = noised_img[y - offset:y + block_h + offset, x - offset:x + block_w + offset]
                #     tmp = median_blur_ch1(blur_block, kernel_size, mode='four')
                #     noised_img[y:y + block_h, x:x + block_w] = tmp[offset:-offset, offset:-offset]
                # else:
                #     blur_block = noised_img[y:y + block_h, x:x + block_w]
                #     noised_img[y:y + block_h, x:x + block_w] = median_blur_ch1(blur_block, kernel_size, mode='four')

                cv2.rectangle(noised_img, (x, y), (x + block_w, y + block_h), (0, 0, 255), 2)

    return noised_img


def median_based_denoise_ch3(noised_img, thresh, block_div, kernel_size):
    assert len(noised_img.shape) == 3
    # img_yuv = cv2.cvtColor(noised_img, cv2.COLOR_BGR2YUV)
    b, g, r = cv2.split(noised_img)
    b = median_based_denoise_ch1(b, thresh, block_div, kernel_size)
    g = median_based_denoise_ch1(g, thresh, block_div, kernel_size)
    r = median_based_denoise_ch1(r, thresh, block_div, kernel_size)
    merged = cv2.merge([b, g, r])
    return merged


# if __name__ == '__main__':
#     # path = '/home/SENSETIME/sunxin/0_data/Set12/08.png'
#     # rgb = cv2.imread(path)
#     # noised_img = random_noise(rgb, 3000)
#     # noised_img = gasuss_noise(rgb)
#     # noised_img = cv2.cvtColor(noised_img, cv2.COLOR_BGR2GRAY)
#
#     path = '/home/SENSETIME/sunxin/0_data/front/resize/1.png'
#     # path = '/home/SENSETIME/sunxin/0_data/BSD300/2092.jpg'
#     noised_img = cv2.imread(path)
#     # noised_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
#
#     noised = copy.deepcopy(noised_img)
#     cv2.putText(noised, 'noised', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#
#     block_div = 8
#     thresh = 9
#     kernel_size = 3
#
#     # input1 = copy.deepcopy(noised_img)
#     # cv2_median = median_blur_ch1(input1, kernel_size, mode='default')
#     # cv2.putText(cv2_median, 'cv2_median', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#     # input2 = copy.deepcopy(noised_img)
#     # our_median = median_blur_ch1(input2, kernel_size, mode='four')
#     # cv2.putText(our_median, 'our_median', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#     # input3 = copy.deepcopy(noised_img)
#     # our_final = median_based_denoise_ch1(input3, thresh=thresh, block_div=block_div, kernel_size=kernel_size)
#     # cv2.putText(our_final, 'our_final', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
#     # output = cv2.hconcat([noised, cv2_median, our_median, our_final])
#
#     input1 = copy.deepcopy(noised_img)
#     cv2_median = cv2.medianBlur(input1, kernel_size)
#     cv2.putText(cv2_median, 'cv2_median', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#     input3 = copy.deepcopy(noised_img)
#     our_final = median_based_denoise_ch3(input3, thresh=thresh, block_div=block_div, kernel_size=kernel_size)
#     cv2.putText(our_final, 'our_final', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
#     output = cv2.hconcat([noised, cv2_median, our_final])
#
#     cv2.namedWindow('result', 0)
#     cv2.imshow('result', output)
#     cv2.waitKey()

def get_hist(list_, depth=255):
    hist = dict()
    for i in range(depth + 1):
        hist.setdefault(i, 0)
    # print(hist)
    for key in list_:
        # print(key)
        if hist.get(key) is not None:
            hist[key] += 1
    return hist


def FastMedianFilter(noised, kernel_size, mode='fast'):
    assert len(noised.shape) == 2
    offset = kernel_size // 2
    noised_padded = np.lib.pad(noised, offset, mode='symmetric')
    # print(noised_padded.shape)
    denoised = copy.deepcopy(noised)
    h, w = noised.shape

    for y in range(h):
        for x in range(w):
            cell = noised_padded[y:y + kernel_size, x:x + kernel_size]
            cnt_thresh = kernel_size * kernel_size // 2 + 1
            median_val = 0
            sum_cnt = 0
            cur_hist = dict()
            # print(x,y)
            if x == 0:
                cur_hist = get_hist(cell.flatten())
                for key, val in cur_hist.items():
                    sum_cnt += val
                    if sum_cnt >= cnt_thresh:
                        median_val = key
                        break
            else:
                pre_col = noised_padded[y:y + kernel_size, x - 1]
                for key in pre_col:
                    cur_hist[key] = cur_hist.get(key, 0) - 1
                    sum_cnt -= 1
                new_col = noised_padded[y:y + kernel_size, x + kernel_size]
                for key in new_col:
                    cur_hist[key] = cur_hist.get(key, 0) + 1
                    sum_cnt += 1
                # print(pre_col.shape, new_col.shape)

                # if sum_cnt < cnt_thresh:
                #     # print(cur_hist)
                #     for key in range(median_val, 256, 1):
                #         # print(cur_hist[key])
                #         sum_cnt += cur_hist[key]
                #         if sum_cnt >= cnt_thresh:
                #             median_val = key
                #             break
                # elif sum_cnt > cnt_thresh:
                #     for key in range(median_val, 1, -1):
                #         sum_cnt -= cur_hist[key]
                #         if sum_cnt <= cnt_thresh:
                #             median_val = key
                #             break
                # else:
                #     pass

            # print(cell.flatten())
            # cur_hist = get_hist(cell.flatten())
            # for key, val in cur_hist.items():
            #     sum_cnt += val
            #     if sum_cnt >= cnt_thresh:
            #         median_val = key
            #         break

            denoised[y, x] = median_val

    return denoised

def gaussian_kernel_2d_opencv(kernel_size = 3,sigma = 0):
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    return np.multiply(kx,np.transpose(ky))

def WeightedMedianFilter(noised, kernel_size):
    assert len(noised.shape) == 2
    offset = kernel_size // 2
    noised_padded = np.lib.pad(noised, offset, mode='symmetric')
    denoised = copy.deepcopy(noised)
    h, w = noised.shape
    for y in range(h):
        for x in range(w):
            cell = noised_padded[y:y + kernel_size, x:x + kernel_size]

            p = denoised[y, x]

            # mid_index = kernel_size * kernel_size // 2
            qs = cell.flatten()
            # qs = qs.pop(mid_index)
            ws = []

            # W = gaussian_kernel_2d_opencv(kernel_size=kernel_size)
            def function_pq(pixel):
                return pixel

            for q in qs:
                fq = function_pq(q)
                fp = function_pq(p)
                dist = np.abs(fp-fq)
                w = np.exp(dist * -1)
                ws.append(w)



            # ws_sum = np.sum(ws)

            # ws_reverse_sorted = np.
            # ws_reverse_sorted = np.
            # ws_sum_k = 0
            # for w in ws:
            #     ws_sum_k += w
            #     if ws_sum_k >=






            denoised[y, x] = median_val

    return denoised


if __name__ == '__main__':
    path = '/home/SENSETIME/sunxin/0_data/Set12/08.png'
    src = cv2.imread(path)
    src = cv2.resize(src, None, fx=0.25, fy=0.25)
    noised_img = random_noise(src, 3000)
    noised = cv2.cvtColor(noised_img, cv2.COLOR_BGR2GRAY)
    # print(noised.flatten())
    # print(get_hist(noised.flatten()))

    denoised = FastMedianFilter(noised, kernel_size=5)
    output = cv2.hconcat([noised, denoised])
    cv2.namedWindow('result', 0)
    cv2.imshow('result', output)
    cv2.waitKey()

    # noised = copy.deepcopy(noised_img)
    # cv2.putText(noised, 'noised', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    #
    # block_div = 8
    # thresh = 9
    # kernel_size = 3
    #
    # # input1 = copy.deepcopy(noised_img)
    # # cv2_median = median_blur_ch1(input1, kernel_size, mode='default')
    # # cv2.putText(cv2_median, 'cv2_median', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    # # input2 = copy.deepcopy(noised_img)
    # # our_median = median_blur_ch1(input2, kernel_size, mode='four')
    # # cv2.putText(our_median, 'our_median', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    # # input3 = copy.deepcopy(noised_img)
    # # our_final = median_based_denoise_ch1(input3, thresh=thresh, block_div=block_div, kernel_size=kernel_size)
    # # cv2.putText(our_final, 'our_final', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    # # output = cv2.hconcat([noised, cv2_median, our_median, our_final])
    #
    # input1 = copy.deepcopy(noised_img)
    # cv2_median = cv2.medianBlur(input1, kernel_size)
    # cv2.putText(cv2_median, 'cv2_median', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    # input3 = copy.deepcopy(noised_img)
    # our_final = median_based_denoise_ch3(input3, thresh=thresh, block_div=block_div, kernel_size=kernel_size)
    # cv2.putText(our_final, 'our_final', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
