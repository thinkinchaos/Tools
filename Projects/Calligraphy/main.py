import copy
import cv2
import math
import numpy as np
import xlwt
import os

book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('calligraphy', cell_overwrite_ok=True)
sheet.write(0, 0, '框序号')
sheet.write(0, 1, '框中心')
sheet.write(0, 2, '字半径')
sheet.write(0, 3, '字重心')
sheet.write(0, 4, '连接向量角度')
sheet.write(0, 5, '连接向量长度')
sheet.write(0, 6, '黑色占比')
sheet.write(0, 7, '最细值')
sheet.write(0, 8, '最粗值')
sheet.write(0, 9, '最细位置')
sheet.write(0, 10, '最粗位置')
sheet.write(0, 11, '收笔方向')
coordinates = []


class Rect(object):
    def __init__(self):
        self.tl = (0, 0)
        self.br = (0, 0)

    def regularize(self):
        pt1 = (min(self.tl[0], self.br[0]), min(self.tl[1], self.br[1]))
        pt2 = (max(self.tl[0], self.br[0]), max(self.tl[1], self.br[1]))
        self.tl = pt1
        self.br = pt2


class DrawRects(object):
    def __init__(self, image, color, thickness=1):
        self.original_image = image
        self.image_for_show = image.copy()
        self.color = color
        self.thickness = thickness
        self.rects = []
        self.current_rect = Rect()
        self.left_button_down = False

    @staticmethod
    def __clip(value, low, high):
        output = max(value, low)
        output = min(output, high)
        return output

    def shrink_point(self, x, y):
        height, width = self.image_for_show.shape[0:2]
        x_shrink = self.__clip(x, 0, width)
        y_shrink = self.__clip(y, 0, height)
        return (x_shrink, y_shrink)

    def append(self):
        self.rects.append(copy.deepcopy(self.current_rect))

    def pop(self):
        rect = Rect()
        if self.rects:
            rect = self.rects.pop()
        return rect

    def reset_image(self):
        self.image_for_show = self.original_image.copy()

    def draw(self):
        for rect in self.rects:
            cv2.rectangle(self.image_for_show, rect.tl, rect.br, color=self.color, thickness=self.thickness)

    def draw_current_rect(self):
        cv2.rectangle(self.image_for_show, self.current_rect.tl, self.current_rect.br, color=self.color,
                      thickness=self.thickness)


def onmouse_draw_rect(event, x, y, flag, draw_rects):
    tmp = []
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append(x)
        coordinates.append(y)
        draw_rects.left_button_down = True
        draw_rects.current_rect.tl = (x, y)
    if draw_rects.left_button_down and event == cv2.EVENT_MOUSEMOVE:
        draw_rects.current_rect.br = draw_rects.shrink_point(x, y)
        draw_rects.reset_image()
        draw_rects.draw()
        draw_rects.draw_current_rect()
    if event == cv2.EVENT_LBUTTONUP:
        draw_rects.left_button_down = False
        draw_rects.current_rect.br = draw_rects.shrink_point(x, y)
        tmp.append(draw_rects.current_rect.br[0])
        tmp.append(draw_rects.current_rect.br[1])
        x = draw_rects.current_rect.br[0]
        y = draw_rects.current_rect.br[1]
        coordinates.append(x)
        coordinates.append(y)
        draw_rects.current_rect.regularize()
        draw_rects.append()
    if (not draw_rects.left_button_down) and event == cv2.EVENT_RBUTTONDOWN:
        draw_rects.pop()
        draw_rects.reset_image()
        draw_rects.draw()


def get_max_contour(contours):
    max_contour = contours[0]
    max_contour_area = 0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > max_contour_area:
            max_contour = contour
            max_contour_area = contour_area
    return max_contour, max_contour_area


def get_skeleton(binary):
    dst = binary.copy()
    num_erode = 0
    while (True):
        if np.sum(dst) == 0:
            break
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dst = cv2.erode(dst, kernel)
        num_erode = num_erode + 1
    skeleton = np.zeros(dst.shape, np.uint8)
    for x in range(num_erode):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dst = cv2.erode(binary, kernel, None, None, x)
        open_dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
        result = dst - open_dst
        skeleton = skeleton + result
    return skeleton


if __name__ == '__main__':
    # 用鼠标手动框，按Ese键结束
    image = cv2.imread('source.png')
    draw_rects = DrawRects(image, (0, 255, 0), 2)
    WIN_NAME = 'draw_rect'
    cv2.namedWindow(WIN_NAME, 0)
    cv2.setMouseCallback(WIN_NAME, onmouse_draw_rect, draw_rects)
    while True:
        cv2.imshow(WIN_NAME, draw_rects.image_for_show)
        key = cv2.waitKey(30)
        if key == 27:
            break
    cv2.destroyWindow(WIN_NAME)
    image1 = copy.deepcopy(image)
    word_ths = []

    assert len(coordinates) % 4 == 0
    rects = []
    for i in range(0, len(coordinates), 4):
        tmp = []
        for j in range(4):
            tmp.append(coordinates[i + j])
        rects.append(tmp)
    for id, rect in enumerate(rects):
        # 对框区域预处理
        img = image[rect[1]:rect[3], rect[0]:rect[2], :]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        word_ths.append(ret)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel1)
        dilated = cv2.dilate(dilated, kernel1)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(dilated, kernel2)
        binary = eroded

        # 提取框和字的一般信息
        center = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)  # 框的中心
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        maxContour, maxContourArea = get_max_contour(contours)
        rect_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        black_ratio = maxContourArea / rect_area * 100
        b_x, b_y, b_w, b_h = cv2.boundingRect(maxContour)
        radius = max(b_w, b_h) // 2  # 字的半径
        mom = cv2.moments(maxContour)
        weight = (int(mom['m10'] / mom['m00']) + rect[0], int(mom['m01'] / mom['m00']) + rect[1])  # 字的重心
        vector = (center[0] - weight[0], center[1] - weight[1])  # 框中心到字重心的向量，即连接向量
        degree = math.degrees(math.atan2(vector[1] - vector[0], 1))  # 连接向量与x轴的角度
        length = math.sqrt(math.pow((center[0] - weight[0]), 2) + math.pow((center[1] - weight[1]), 2))  # 连接向量的长度

        # 提取字的粗细信息
        split_factor = 20
        tmp_w, tmp_h = b_w / split_factor, b_h / split_factor
        tmp_rect_area = tmp_w * tmp_h
        tmp_ratio_max = 0
        tmp_ratio_min = 200
        thick_x, thick_y, thick_size = 0, 0, 10
        thin_x, thin_y, thin_size = 0, 0, 2
        for i in range(split_factor):
            tmp_x1, tmp_y1 = b_x + i * split_factor, b_y + i * split_factor
            tmp_binary = binary[int(tmp_y1):int(tmp_y1 + tmp_h), int(tmp_x1):int(tmp_x1 + tmp_w)]
            tmp_contours, _ = cv2.findContours(tmp_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(tmp_contours) == 0:
                continue
            tmp_max_contour, tmp_max_contour_area = get_max_contour(tmp_contours)
            tmp_ratio = tmp_max_contour_area / tmp_rect_area
            if tmp_ratio > tmp_ratio_max:
                tmp_ratio_max = tmp_ratio
                thick_x = tmp_x1 + rect[0]
                thick_y = tmp_y1 + rect[1]
                thick_size = math.sqrt(tmp_max_contour_area)
            if tmp_ratio < tmp_ratio_min:
                tmp_ratio_min = tmp_ratio
                thin_x = tmp_x1 + rect[0]
                thin_y = tmp_y1 + rect[1]
                thin_size = math.sqrt(tmp_max_contour_area)
        if thin_x == 0 or thin_y == 0:
            thin_x, thin_y = b_x + b_w // 4, b_y + b_h // 4
        if thick_x == 0 or thick_y == 0:
            thick_x, thick_y = b_x + (b_w // 4 * 3), b_y + (b_h // 4 * 3)
        thick_pt = (thick_x, thick_y)
        thin_pt = (thin_x, thin_y)
        if thin_size == 0:
            thin_size = 2
        if thin_size > thick_size:
            thin_size = thick_size

        # 提取字的手笔方向
        sk = get_skeleton(binary)[b_y:b_y + b_h, b_x:b_x + b_w]
        sk_h, sk_w = sk.shape
        factor = 20
        last_y0 = sk_h // factor * (factor - 1)
        sk_bottom = sk[last_y0:, :]
        bottom_contours, _ = cv2.findContours(sk_bottom, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        max_y = 0
        last_x, last_y = 0, 0
        last_contour = bottom_contours[0]
        for bottom_contour in bottom_contours:
            for xy in bottom_contour:
                if xy[0][1] > max_y:
                    max_y = xy[0][1]
                    last_contour = bottom_contour
                    last_x, last_y = xy[0][0], xy[0][1]
        try:
            mom_ = cv2.moments(last_contour)
            last_mid_x, last_mid_y = int(mom_['m10'] / mom_['m00']), int(mom_['m01'] / mom_['m00'])  # 收笔的重心
        except:
            last_mid_x, last_mid_y = last_x - 10, last_y - 10
        last_vector = [last_mid_x - last_x, last_mid_y - last_y]
        x_vector = [1, 0]
        dot = last_vector[0] * x_vector[0] + last_vector[1] * x_vector[1]
        det = last_vector[0] * x_vector[1] - last_vector[1] * x_vector[0]
        theta = np.arctan2(det, dot)
        theta = theta if theta > 0 else 2 * np.pi + theta
        last_degree = theta * 180 / np.pi  # 在图像坐标系下，重心向量相对于中心向量在顺时针方向的夹角

        # 打印信息
        print('\n################################### ID:', id + 1)
        print('框中心：', '{}, {}'.format(center[0], center[1]))
        print('字半径：', radius)
        print('字重心：', '{}, {}'.format(weight[0], weight[1]))
        print('连接向量角度：', np.round(degree, 2))
        print('连接向量长度：', np.round(length, 2))
        print('黑色占比：', np.round(black_ratio, 2))
        print('最细值：', np.round(thin_size, 2))
        print('最粗值：', np.round(thick_size, 2))
        print('最细位置：', '{}, {}'.format(thin_pt[0], thin_pt[1]))
        print('最粗位置：', '{}, {}'.format(thick_pt[0], thick_pt[1]))
        print('收笔方向：', np.round(last_degree))

        sheet.write(id + 1, 0, str(id + 1))
        sheet.write(id + 1, 1, '{}, {}'.format(center[0], center[1]))
        sheet.write(id + 1, 2, str(radius))
        sheet.write(id + 1, 3, '{}, {}'.format(weight[0], weight[1]))
        sheet.write(id + 1, 4, str(np.round(degree, 2)))
        sheet.write(id + 1, 5, str(np.round(length, 2)))
        sheet.write(id + 1, 6, str(np.round(black_ratio, 2)))
        sheet.write(id + 1, 7, str(np.round(thin_size, 2)))
        sheet.write(id + 1, 8, str(np.round(thick_size, 2)))
        sheet.write(id + 1, 9, '{}, {}'.format(thin_pt[0], thin_pt[1]))
        sheet.write(id + 1, 10, '{}, {}'.format(thick_pt[0], thick_pt[1]))
        sheet.write(id + 1, 11, str(np.round(last_degree)))

        # 可视化信息
        cv2.rectangle(image1, (rect[0], rect[1]), (rect[2], rect[3]), color=(0, 255, 0), thickness=2)  # 绿框，原始框
        cv2.rectangle(image1, (b_x + rect[0], b_y + rect[1]),
                      (b_x + b_w + rect[0], b_y + b_h + rect[1]), color=(255, 0, 0), thickness=2)  # 蓝框，字框
        cv2.circle(image1, thick_pt, 4, color=(0, 0, 0), thickness=-1)  # 黑点，最粗处
        cv2.circle(image1, thin_pt, 4, color=(255, 255, 255), thickness=-1)  # 白点，最细处
        cv2.line(image1, center, weight, color=(0, 0, 255), thickness=2)  # 红线，字重心与框中心的连接向量
        cv2.line(image1, (last_mid_x + b_x + rect[0], last_mid_y + b_y + rect[1] + last_y0),
                 (b_x + rect[0] + last_x, b_y + rect[1] + last_y + last_y0), color=(0, 0, 255), thickness=2)  # 红线，收笔向量

    ###################################################################################################################

    # 分割处字前景
    gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary1 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask = ~(cv2.inRange(hsv, lower_red, upper_red).astype('bool'))
    h_, w_ = mask.shape
    for i in range(h_):
        for j in range(w_):
            if j < 100:
                mask[i][j] = False
            if j > w_ - 100:
                mask[i][j] = False
    binary2 = binary1 * mask

    # 制作背景颜色的画布
    bgr_data = np.float32(image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 3
    ret, label, center = cv2.kmeans(bgr_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    label = list(label.squeeze())
    dicts = {}
    for lab in label:
        if lab not in dicts.keys():
            dicts[lab] = label.count(lab)
    max_lab_num = 0
    most_lab = 0
    for k, v in dicts.items():
        if max_lab_num < v:
            max_lab_num = v
            most_lab = k
    most_color = center[most_lab]
    image2 = np.zeros(image.shape, dtype='uint8')
    for i in range(3):
        image2[:, :, i] = int(most_color[i])
    image3 = copy.deepcopy(image2)

    # 对前景形态学处理
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(binary2, kernel1)  # 消除噪点
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    dilated = cv2.dilate(eroded, kernel2)
    dilated = cv2.dilate(dilated, kernel2)  # 横向融合
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    dilated = cv2.dilate(dilated, kernel3)
    dilated = cv2.dilate(dilated, kernel3)  # 纵向融合

    # 面积筛选处不符合条件的前景
    d_contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    d_contours2 = []
    for d_contour in d_contours:
        area = cv2.contourArea(d_contour)
        if area >= 2000:
            d_contours2.append(d_contour)

    # 画黑团
    centers = []
    weights = []
    max_bk_ratio, min_bk_ratio = 0, 10
    max_bk_contour, min_bk_contour = d_contours2[0], d_contours2[0]
    for d_contour in d_contours2:
        mom = cv2.moments(d_contour)
        x_, y_ = int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])
        weights.append([x_, y_])
        b_x, b_y, b_w, b_h = cv2.boundingRect(d_contour)
        centers.append([b_x + b_w // 2, b_y + b_h // 2])
        radius = max(b_w, b_h) // 2  # 字的半径
        cv2.circle(image2, (x_, y_), radius, (0, 0, 255), 2)
        contour_area = cv2.contourArea(d_contour)
        circle_area = 3.14 * radius * radius
        bk_ratio = contour_area / circle_area
        if bk_ratio > max_bk_ratio:
            max_bk_ratio = bk_ratio
            max_bk_contour = d_contour
        if bk_ratio < min_bk_ratio:
            min_bk_ratio = bk_ratio
            min_bk_contour = d_contour
        cv2.drawContours(image3, [d_contour], -1, (0 + 255 * bk_ratio, 0 + 255 * bk_ratio, 0 + 255 * bk_ratio), -1)

    # 提取字的粗细信息(min)
    b_x, b_y, b_w, b_h = cv2.boundingRect(min_bk_contour)
    split_factor = 20
    tmp_w, tmp_h = b_w / split_factor, b_h / split_factor
    tmp_rect_area = tmp_w * tmp_h
    tmp_ratio_max = 0
    tmp_ratio_min = 200
    thick_x, thick_y, thick_size = 0, 0, 10
    thin_x, thin_y, thin_size = 0, 0, 2
    for i in range(split_factor):
        tmp_x1, tmp_y1 = b_x + i * split_factor, b_y + i * split_factor
        tmp_binary = binary2[int(tmp_y1):int(tmp_y1 + tmp_h), int(tmp_x1):int(tmp_x1 + tmp_w)]
        tmp_contours, _ = cv2.findContours(tmp_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(tmp_contours) == 0:
            continue
        tmp_max_contour, tmp_max_contour_area = get_max_contour(tmp_contours)
        tmp_ratio = tmp_max_contour_area / tmp_rect_area
        if tmp_ratio < tmp_ratio_min:
            tmp_ratio_min = tmp_ratio
            thin_x = tmp_x1
            thin_y = tmp_y1
            thin_size = math.sqrt(tmp_max_contour_area)
    if thin_size == 0:
        thin_size = 2
    if thin_x == 0 or thin_y == 0:
        thin_x, thin_y = b_x + b_w // 4, b_y + b_h // 4
    thin_pt = (thin_x, thin_y)

    # 提取字的粗细信息(max)
    b_x, b_y, b_w, b_h = cv2.boundingRect(max_bk_contour)
    split_factor = 20
    tmp_w, tmp_h = b_w / split_factor, b_h / split_factor
    tmp_rect_area = tmp_w * tmp_h
    tmp_ratio_max = 0
    tmp_ratio_min = 200
    thick_x, thick_y, thick_size = 0, 0, 10
    thin_x, thin_y, thin_size = 0, 0, 2
    for i in range(split_factor):
        tmp_x1, tmp_y1 = b_x + i * split_factor, b_y + i * split_factor
        tmp_binary = binary2[int(tmp_y1):int(tmp_y1 + tmp_h), int(tmp_x1):int(tmp_x1 + tmp_w)]
        tmp_contours, _ = cv2.findContours(tmp_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(tmp_contours) == 0:
            continue
        tmp_max_contour, tmp_max_contour_area = get_max_contour(tmp_contours)
        tmp_ratio = tmp_max_contour_area / tmp_rect_area
        if tmp_ratio > tmp_ratio_max:
            tmp_ratio_max = tmp_ratio
            thick_x = tmp_x1
            thick_y = tmp_y1
            thick_size = math.sqrt(tmp_max_contour_area)
    if thick_x == 0 or thick_y == 0:
        thick_x, thick_y = b_x + (b_w // 4 * 3), b_y + (b_h // 4 * 3)
    thick_pt = (thick_x, thick_y)

    # 画粗细点
    cv2.circle(image2, thick_pt, 20, (0, 0, 255), -1)
    cv2.circle(image2, thin_pt, 20, (0, 0, 255), -1)

    # 画中心点连线，蓝色
    centers = np.array(centers)
    centers = centers[np.argsort(centers[:, 1]), :]  # 按y坐标排序
    xs = np.array([w[0] for w in centers]).astype('float32')
    criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)
    k = 6
    ret, label, center = cv2.kmeans(xs, k, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)
    for line_x in center:
        cv2.line(image2, (line_x, 50), (line_x, h_ - 50), (0, 255, 255), 2)
    lines = []
    for lab in range(k):
        line = []
        for i in range(len(centers)):
            if label[i] == lab:
                line.append(centers[i])
        lines.append(line)
    for line in lines:
        for i in range(len(line) - 1):
            cv2.line(image2, tuple(line[i]), tuple(line[i + 1]), (255, 0, 0), 2)

    # 画重心连线，绿色
    weights = np.array(weights)
    weights = weights[np.argsort(weights[:, 1]), :]  # 按y坐标排序
    xs = np.array([w[0] for w in weights]).astype('float32')
    criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)
    k = 6
    ret, label, center = cv2.kmeans(xs, k, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)
    for line_x in center:
        cv2.line(image2, (line_x, 50), (line_x, h_ - 50), (0, 255, 255), 2)
    lines = []
    for lab in range(k):
        line = []
        for i in range(len(weights)):
            if label[i] == lab:
                line.append(weights[i])
        lines.append(line)
    for line in lines:
        for i in range(len(line) - 1):
            cv2.line(image2, tuple(line[i]), tuple(line[i + 1]), (0, 255, 0), 2)

    # 显示图片，按任意键结束
    cv2.namedWindow('Visualize extraction', 0)
    cv2.imshow('Visualize extraction', image1)
    cv2.namedWindow('Output Image 1', 0)
    cv2.imshow('Output Image 1', image2)
    cv2.namedWindow('Output Image 2', 0)
    cv2.imshow('Output Image 2', image3)
    cv2.waitKey()

    # 保存图片文件
    cv2.imwrite('Visualize.jpg', image1)
    cv2.imwrite('OutputImage1.jpg', image2)
    cv2.imwrite('OutputImage2.jpg', image3)
    # 保存Excel文件
    save_file = 'information.xls'
    if os.path.exists(save_file):
        os.remove(save_file)
    else:
        book.save(save_file)
