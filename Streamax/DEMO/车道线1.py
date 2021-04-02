import copy
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont

showtxt_in_frame = False
show_result = True
write_result = False

color = (0, 255, 255)

lines_certain_frame = []
lines_all_frames = []
count = 0
for line in open("../mydata/result2.txt"):
    if line != '\n':
        if line[0] != 'f':
            list1 = eval(line)
            lines_certain_frame.append(list1)
            count += 1

        if count == 4:
            tmp = copy.deepcopy(lines_certain_frame)
            lines_all_frames.append(tmp)

            lines_certain_frame.clear()
            count = 0

cap = cv2.VideoCapture('../mydata/test-env.avi')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frameWidth, frameHeight)
total_frame_num_VIDEO = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter('../mydata/test-env.avi.avi', cv2.VideoWriter_fourcc(*"MJPG"), 25, size)


def show_text(image, variable_name, variable_val, tl_x=10, tl_y=10, tab=50, color=color, thick=2):
    cv2.putText(image, variable_name + ': ' + str(variable_val), (tl_x, tl_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                thick)


def draw_line_on_frame(rgba_image, pt1, pt2):

    # line_image_rgba = Image.new('RGBA', rgba_image.size, (0, 0, 0, 100))
    draw = ImageDraw.Draw(rgba_image)

    draw.line((pt1[0], pt1[1], pt2[0], pt2[1]), (255, 255, 0, 80), 15)

    # new_image = Image.alpha_composite(rgba_image, line_image_rgba)

    return rgba_image

total_frame_num_TXT = len(lines_all_frames)

for frame_idx in range(total_frame_num_TXT):
    ret, frame = cap.read()

    # frame = np.zeros((500,500,3),np.uint8)


    # src_frame = copy.deepcopy(frame)
    line_image = np.zeros(frame.shape, np.array(frame).dtype)
    frame_pil_rgba = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')

    # line_image_pil_rgba = Image.fromarray(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)).convert('RGBA')



    if showtxt_in_frame:
        show_text(frame, 'fps', fps, 0, 30)
        show_text(frame, 'frameWidth', frameWidth, 200, 30)
        show_text(frame, 'frameHeight', frameHeight, 600, 30)
        show_text(frame, 'total_frame_num_VIDEO', total_frame_num_VIDEO, 0, 60)
        show_text(frame, 'total_frame_num_TXT', total_frame_num_TXT, 600, 60)
        show_text(frame, 'frame_idx', frame_idx, 0, 90)

    line_num = len(lines_all_frames[frame_idx])

    for line_idx in range(line_num):
        if showtxt_in_frame:
            show_text(frame, 'line_idx', line_idx, 200, 120 + line_idx * 30)
            show_text(frame, 'point_num', len(lines_all_frames[frame_idx][line_idx]), 400, 120 + line_idx * 30)

        point_cur = ()
        point_pre = ()
        point_idx = 0

        for point in lines_all_frames[frame_idx][line_idx]:
            pt = (int(point[0]), int(point[1]))

            if showtxt_in_frame:
                if point_idx == 0:
                    show_text(frame, 'start_x', pt[0], 800, 120 + line_idx * 30)
                    show_text(frame, 'start_y', pt[1], 1100, 120 + line_idx * 30)

            point_cur = copy.deepcopy(pt)

            if point_idx > 0:
                thick = 15

                result_pil = draw_line_on_frame(frame_pil_rgba, point_pre, point_cur)
                # cv2.line(line_image, point_pre, point_cur, color, thick)
                # line_image_pil_rgba.text((100, 100), "AAA", font=ImageFont.truetype('C:\Windows\Fonts\STXINGKA.TTF', 36), fill=(0, 0, 0, 50))

            point_idx += 1

            point_pre = copy.deepcopy(point_cur)

    # print('\n')
    # show_image = np.zeros(frame.shape, np.uint8)
    # height, weight, channels = line_image.shape
    # for row in range(height):  # 遍历每一行
    #     for col in range(weight):  # 遍历每一列
    #         for channel in range(channels):
    #             if line_image[row][col][channel] != 0:
    #                 frame[row][col][channel] = (line_image[row][col][channel] + frame[row][col][channel])/2

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # dilated = cv2.dilate(line_image, kernel)  # 膨胀图像
    # eroded = cv2.erode(line_image, kernel)  # 腐蚀图像


    # show_image = cv2.addWeighted(frame, 0.85, line_image, 0.15, 0)
    # show_image = cv2.add(frame, line_image)



    result_cv2 = cv2.cvtColor(np.asarray(result_pil), cv2.COLOR_RGBA2BGRA)
    # import time
    if show_result:
        # result_pil.show()
        # time.sleep(0.01)
        # result_pil.close()


        # time.sleep(0.01)
        cv2.namedWindow('show', 0)
        cv2.imshow('show', result_cv2)
        cv2.waitKey(1)

    if write_result:
        result_pil.save(str(frame_idx) + '.png')



        # out.write(frame)


