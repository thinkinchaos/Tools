import json
import os
# import warnings
from PIL import Image
# import yaml
from labelme import utils
import base64
from pathlib import Path


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input')
    # parser.add_argument('--output')
    # args = parser.parse_args()
    # in_dir = args.input
    # out_dir = args.output

    in_dir = 'D:/DATA/shenzhen_rsycnzebra_crossing_done_527/full_json_fast/save'
    out_dir = 'D:/DATA/shenzhen_rsycnzebra_crossing_done_527/full_json_fast/vis'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    json_paths = [i for i in Path(in_dir).glob('*.json')]

    for json_path in json_paths:

        file_name = json_path.name[:-5]
        file_dir = json_path.parent

        data = json.load(open(str(json_path)))

        try:
            imageData = data.get('imageData')
            if not imageData:
                image_path = os.path.join(file_dir, data['imagePath'])
                with open('D:/1.png', 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
                img = utils.img_b64_to_arr(imageData)
                lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])
                # if 1 in lbl:
                #     print('ok')
                # print(lbl)
                # import numpy as np
                # lbl = np.where(lbl!=0, 255,0)
                # # print(lbl)
                # vis = Image.fromarray(lbl)
                # from matplotlib import pyplot as plt
                # plt.imread(vis)
                # plt.show()
                # import cv2
                # cv2.imshow('ss', lbl)
                # cv2.waitKey()
                Image.fromarray(lbl).save(os.path.join(out_dir, '{}.png'.format(file_name)))
                print(len(Image.fromarray(lbl).split()))

                # captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
                # lbl_viz = ML.DEMO.draw_label(lbl, img, captions)
                # PIL.Image.fromarray(img).save(os.path.join(out_dir, '{}.png'.format(filename)))
                # PIL.Image.fromarray(lbl).save(osp.join(out_dir, '{}_mask.png'.format(filename)))
                # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.jpg'.format(filename)))
                # with open(os.path.join(out_dir, 'label_names.txt'), 'w') as f:
                #     for lbl_name in lbl_names:
                #         f.write(lbl_name + '\n')
                # warnings.warn('info.yaml is being replaced by label_names.txt')
                # info = dict(label_names=lbl_names)
                # with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                #     yaml.safe_dump(info, f, default_flow_style=False)
        except OSError:
            pass
        continue


if __name__ == '__main__':
    main()
