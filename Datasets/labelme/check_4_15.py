check = '//192.168.133.14/shenzhen_rsycn/adas_parsing_data_1911_b/check'
repeat = 'C:/Users/Gigabyte/Desktop/adas_parsing_data_0409_to0410'
from pathlib import Path

# c=Path(check)
# p=Path(repeat)
# check_dirs = [i for i in c.iterdir() if i.is_dir]
# repeat_dirs = [i for i in p.iterdir() if i.is_dir]

img_paths = [i for i in Path(repeat).rglob('*.jpg')]
json_paths = [i for i in Path(repeat).rglob('*.json')]
# images = c.rglob('*.jpg')
# print(len(jpgs) + len(jsons))
import cv2

# img_names = []
# for img_path in img_paths:
#     img_name = img_path.name
#     img_names.append(img_name)
# 
# for idx, img_name in enumerate(img_names):
#     # print(img_name)
#     if img_names.count(img_name) > 1:
#         print(img_paths[idx])
        
json_names = []
for json_path in json_paths:
    json_name = json_path.name
    json_names.append(json_name)

repeat_json_names=set()
for idx, json_name in enumerate(json_names):
    # repeat_json_path_group_tmp = []
    if json_names.count(json_name) > 1:
        repeat_json_names.add(json_name)




for repeat_json_name in repeat_json_names:
    json_paths_this_name_in_check = [str(i) for i in Path(check).rglob(repeat_json_name)]
    print(json_paths_this_name_in_check)

#     # repeat_json_path_group_tmp = []
#     if json_names.count(json_name) > 1:
#         repeat_json_names.add(json_name)


# import shutil
#
# for repeat_json_name in repeat_json_names:
#     repeat_json_paths_this_name = [i for i in Path(repeat).rglob(repeat_json_name)]
#     # print(repeat_json_paths_this_name)
#     with open(str(file-tools), 'r') as f:
#         dict = json.load(f)
#         image_name = dict['imagePath']
#         if '.jpg'in image_name:
#             shutil.copy(str(file-tools), new_path)
#         else:
#             continue

# print(repeat_json_names)
# img_paths = [i for i in Path(repeat).rglob('*.jpg')]
# json_paths = [i for i in Path(repeat).rglob('*.json')]

# img_names = []
# for img_path in imgs:
#     img_name = img_path.name
#     img_names.append(img_name)
#
# for img_name in img_names:
#     # print(img_name)
#     if img_names.count(img_name) > 1:
#         print(img_name)

import os

# for root, dirs, files in os.walk(repeat):
#     if os.path.isfile(fname)
#     # for rqcodeFile in files:
#     print(str(files))

# print(check_dirs)
# from pathlib import Path
# import shutil
# import cv2
# # root = '192.168.133.14/ai_lab/gaoxiang/dataset/train_dataset/adas_parsing_data_0409_to0410/'
# json_paths = []
# image_paths = []
#
# file-tools = open("./repeat_log.txt", 'r', encoding='utf-8')
# for line in file-tools.readlines():
#
#     # check_file_path = root + line
#     # check_file_path = str(check_file_path)
#     # check_file_path = repr(check_file_path)
#     # line = line.replace("\\", "/")
#     # line = 'line
#     # check_file_path = str(check_file_path)
#     line = eval(line)
#     # check_file_path.strip('\n')
#
#     # print(line)
#
#     if line[:-1] == 'g':
#         print(line)
#     #     json_paths.append(check_file_path)
#     # suffix = check_file_path[-5:]
#     # if suffix == 'json':
#     #     print(suffix)
#         # img = cv2.imread(check_file_path)
#         # cv2.imshow("s", img)
#         # cv2.waitKey()
#     #     image_paths.append(check_file_path)
#
#     # print(check_file_path)
#
#     # line=str(line)
#     # line = line.strip('adas_parsing_data_0409_to0410')
#     # print(line)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # dst_path_images = 'C:\\Users\\Gigabyte\\Desktop\\images'
# # dst_path_jsons = 'C:\\Users\\Gigabyte\\Desktop\\jsons'
# # dst_path_repeats = 'C:\\Users\\Gigabyte\\Desktop\\repeats'
# #
# # src_dirs = [i for i in Path(src_path).iterdir() if i.is_dir]
# # processed_file_names = []
# #
# #
# # def process_files(processed_file_names, files):
# #     for file-tools in files:
# #         file_name = file-tools.name
# #         file_suffix = file-tools.suffix
# #         processed_file_names.append(file_name)
# #         if file_name in processed_file_names:
# #             shutil.copy(str(file-tools), dst_path_repeats)
# #         else:
# #             if file_suffix == '.json':
# #                 shutil.copy(str(file-tools), dst_path_jsons)
# #             elif file_suffix == '.jpg':
# #                 shutil.copy(str(file-tools), dst_path_images)
# #         return processed_file_names
# #
# #
# # for _dir in src_dirs:
# #     sub_dirs = [i for i in Path(str(_dir)).iterdir() if i.is_dir()]
# #
# #     sub_sub_dirs = [i for i in Path(str(sub_dirs)).iterdir() if i.is_dir()]
#         # for dir_tmp in sub_sub_dirs:
#
# #
#     # if len(sub_dirs) == 0:
#     #     files1 = [i for i in Path(str(_dir)).iterdir() if i.is_file()]
#     #     processed_file_names = process_files(processed_file_names, files1)
# #
# #     elif len(sub_dirs) > 0:
# #         sub_sub_dirs = [i for i in Path(str(sub_dirs)).iterdir() if i.is_dir()]
# #         for dir_tmp in sub_sub_dirs:
# #             files2 = [i for i in Path(str(dir_tmp)).iterdir() if i.is_file()]
# #             processed_file_names = process_files(processed_file_names, files2)
#
#
