import os
import os.path
import linecache as lc
import numpy as np
import random
import cv2

train_data_path = 'G:/Train_Data'
img_path = train_data_path + '/DataSet'
somking_img_path = 'G:/Train_Data/Sample'
somking_labelimg_fodler = "G:/Train_Data/face_segment/mid"
IMG_W = 1280
IMG_H = 720


class Rect:
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.width = w
        self.high = h

    def aprint(self):
        print(self.x, self.y, self.width, self.high)


def ReadPTS(file_path):
    if not os.path.exists(file_path):
        return []
    line_list = lc.getlines(file_path)
    if len(line_list) == 0:
        print(file_path)
        return []
    begin = line_list.index("{\n")
    end = line_list.index("}\n")
    ret = []
    for i in range(begin + 1, end):
        line = line_list[i]
        tmp = [int(i.split(".")[0]) for i in line.split()]
        if len(tmp) == 0:
            continue
        ret.append(tmp)
    return ret


def ReadFaceRegion(point_list):
    if len(point_list) == 0:
        return
    list_x = [p[0] for p in point_list]
    list_y = [p[1] for p in point_list]
    min_x = min(list_x)
    min_y = min(list_y)
    width = max(list_x) - min_x
    high = max(list_y) - min_y
    return Rect(min_x, min_y, width, high)


def SomkingDetectRegion(point_list):
    if len(point_list) == 0:
        return
    r_face = ReadFaceRegion(point_list)
    if r_face.width == 0 or r_face.high == 0:
        return
    r_face.high = r_face.high - (point_list[27][1] - r_face.y)  # 截取至最低点
    r_face.y = point_list[27][1]

    if IMG_H < r_face.y + r_face.high:
        r_face.high = IMG_H - r_face.y
    return r_face


def DataPathTraverse(somking_path):
    dict = {}
    for data_folder in os.listdir(somking_path):
        for day_folder in os.listdir(os.path.join(somking_path, data_folder)):
            path = os.path.join(somking_path, data_folder, day_folder)
            if os.path.isdir(path):
                dict[day_folder] = path
    print(len(dict))
    return dict


def PossLabel(folder):
    file_list = []
    for file in os.listdir(folder + "/1"):
        name = os.path.splitext(file)
        if name[1] != '.jpg':
            continue
        file_list.append(file)
    return file_list


def NegatLabel(folder):
    file_list = []
    for file in os.listdir(folder + "/2"):
        name = os.path.splitext(file)
        if name[1] != '.jpg':
            continue
        file_list.append(file)

    # if os.path.exists(folder + "/2"):
    #     for file-tools in os.listdir(folder + "/2"):
    #         name = os.path.splitext(file-tools)
    #         if name[1] != '.jpg':
    #             continue
    #         file_list.append(file-tools)
    return file_list


def DisturbanceRect(rect, num):
    if type(rect) != Rect:
        return []
    disx_size = rect.width / 10
    x_rarray = list(np.random.uniform(-disx_size, disx_size, size=num - 1))
    dixy_size = rect.high / 10
    y_rarray = list(np.random.uniform(-dixy_size, dixy_size, size=num - 1))

    ret = [rect]
    # rect.print()
    for i in range(num - 1):
        rect_tmp = Rect()

        rect_tmp.x = rect.x + x_rarray[i]
        if rect_tmp.x + rect.width > IMG_W:
            rect_tmp.width = IMG_W - rect_tmp.x
        else:
            rect_tmp.width = rect.width

        rect_tmp.y = rect.y + y_rarray[i]
        if rect_tmp.y + rect.high > IMG_H:
            rect_tmp.high = IMG_H - rect_tmp.y
        else:
            rect_tmp.high = rect.high

        ret.append(rect_tmp)
        # rect_tmp.print()
    return ret


def CreatLabelTxt(somking_path, label_fodler, ret_path):
    dict = DataPathTraverse(somking_path)

    labelimg_list = PossLabel(label_fodler)
    label_folder_list = [dict[i.split('(')[0]] for i in labelimg_list]
    count = 0
    f = open(ret_path + '/posslabe.txt', 'w')
    for i in range(len(label_folder_list)):
        folder = label_folder_list[i]
        pts_file = folder + '/pts/' + os.path.splitext(labelimg_list[i])[0] + '.pts'
        img_file = folder + '/image/' + labelimg_list[i]
        if not os.path.exists(pts_file) or not os.path.exists(img_file):
            print(pts_file)
            continue
        r_smoking = SomkingDetectRegion(ReadPTS(pts_file))
        rlist_somking = DisturbanceRect(r_smoking, 5)
        for r in rlist_somking:
            tmp = img_file + "\t%d\t%d\t%d\t%d\n" % (r.x, r.y, r.width, r.high)
            f.write(tmp)
        count += 1
        # print(count)
    f.close()

    labelimg_list = NegatLabel(label_fodler)
    random.shuffle(labelimg_list)
    random_postion = int(len(labelimg_list) * 0.5)  # 确定分割位置
    org_img_list = labelimg_list[:random_postion]
    label_folder_list = [dict[i.split('(')[0]] for i in org_img_list]
    count = 0
    f = open(ret_path + '/nonlabel.txt', 'w')
    for i in range(len(label_folder_list)):
        folder = label_folder_list[i]
        pts_file = folder + '/pts/' + os.path.splitext(org_img_list[i])[0] + '.pts'
        img_file = folder + '/image/' + org_img_list[i]
        if not os.path.exists(pts_file) or not os.path.exists(img_file):
            print(pts_file)
            continue
        r_smoking = SomkingDetectRegion(ReadPTS(pts_file))
        f.write(img_file + "\t%d\t%d\t%d\t%d\n" % (r_smoking.x, r_smoking.y, r_smoking.width, r_smoking.high))
        count += 1
        # print(count)

    rand_img_list = labelimg_list[random_postion:int(1.3 * random_postion)]
    label_folder_list = [dict[i.split('(')[0]] for i in rand_img_list]
    for i in range(len(label_folder_list)):
        folder = label_folder_list[i]
        pts_file = folder + '/pts/' + os.path.splitext(org_img_list[i])[0] + '.pts'
        img_file = folder + '/image/' + org_img_list[i]
        if not os.path.exists(pts_file) or not os.path.exists(img_file):
            print(pts_file)
            continue
        r_face = ReadFaceRegion(ReadPTS(pts_file))
        r_face.x = random.randint(0, IMG_W - r_face.width)
        r_face.y = random.randint(0, IMG_H - r_face.high)
        f.write(img_file + "\t%d\t%d\t%d\t%d\n" % (r_face.x, r_face.y, r_face.width, r_face.high))
        count += 1
    f.close()
    print(count)


def RandomPickFormFolder():
    return


if __name__ == '__main__':
    file_path = "./test-env.pts"
    CreatLabelTxt(somking_img_path, somking_labelimg_fodler, ".")

    # test-env = ReadPTS(file_path)
    # print(test-env)
    # print(len(test-env))
    # ReadFaceRegion(test-env).print()
    # SomkingDetectRegion(test-env).print()
    # print(DataPathTraverse("G:/Train_Data/Sample"))
    # print(PoseLabel(somking_labelimg_fodler))
