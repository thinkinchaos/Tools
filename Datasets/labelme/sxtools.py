def mkdir(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)

def batch_labelme_to_mask(labelme_dir, out_dir):

    import json
    import os
    # import warnings
    from PIL import Image
    # import yaml
    from labelme import utils
    import base64
    from pathlib import Path


    def convert(in_dir, out_dir):
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--input')
        # parser.add_argument('--output')
        # args = parser.parse_args()
        # in_dir = args.input
        # out_dir = args.output



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
                    with open(image_path, 'rb') as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode('utf-8')
                    img = utils.img_b64_to_arr(imageData)
                    lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])
                    Image.fromarray(lbl).save(os.path.join(out_dir, '{}.png'.format(file_name)))

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



    import shutil
    out_img_dir = os.path.join(out_dir, 'DEMO')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    # convert(in_dir, out_dir)

    print(labelme_dir[:-10])
    for img in Path(labelme_dir[:-10]).rglob('*.jpg'):
        shutil.copy(str(img), out_img_dir)


def append_obj_in_xml(xmlfile, obj_info):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    element = ET.Element('object')

    sub_element1 = ET.Element('name')
    sub_element1.text = obj_info[1]
    element.append(sub_element1)

    sub_element2 = ET.Element('bndbox')
    xmin = ET.Element('xmin')
    xmin.text = str(obj_info[2])
    ymin = ET.Element('ymin')
    ymin.text = str(obj_info[3])
    xmax = ET.Element('xmax')
    xmax.text = str(obj_info[4])
    ymax = ET.Element('ymax')
    ymax.text = str(obj_info[5])
    sub_element2.append(xmin)
    sub_element2.append(ymin)
    sub_element2.append(xmax)
    sub_element2.append(ymax)

    element.append(sub_element2)
    root.append(element)
    tree.write(xmlfile, encoding='utf-8', xml_declaration=True)

def get_voc_classes(ann_dir):
    import xml.etree.ElementTree as ET
    names=set()
    from pathlib import Path
    for xml_file in Path(ann_dir).glob('*.xml'):
        parser = ET.parse(str(xml_file))
        xml_root = parser.getroot()
        # names_this_xml=[]
        for obj in xml_root.findall('object'):
            name = obj.find('name').text
            names.add(name)
    print(names)


def get_voc_kmeans_anchor_ratios(xml_dir):
    import glob
    import xml.etree.ElementTree as ET

    import numpy as np

    from ML.kmeans import kmeans, avg_iou

    def load_dataset(path):
        dataset = []
        for xml_file in glob.glob("{}/*xml".format(path)):
            tree = ET.parse(xml_file)

            height = int(tree.findtext("./size/height"))
            width = int(tree.findtext("./size/width"))
            # if height==0:
            # 	print(xml_file)
            # print(height, width)

            for obj in tree.iter("object"):
                xmin = int(obj.findtext("bndbox/xmin")) / width
                ymin = int(obj.findtext("bndbox/ymin")) / height
                xmax = int(obj.findtext("bndbox/xmax")) / width
                ymax = int(obj.findtext("bndbox/ymax")) / height

                dataset.append([xmax - xmin, ymax - ymin])

        return np.array(dataset)

    CLUSTERS = 5
    data = load_dataset(xml_dir)
    times = 20
    best_acc = 0
    best_ratio = np.zeros(CLUSTERS)
    for i in range(times):
        out = kmeans(data, k=CLUSTERS)
        acc = avg_iou(data, out)

        print('acc:%.3f' % acc)
        print("Boxes:\n {}".format(out))

        ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
        print("Ratios:\n {}".format(sorted(ratios)))

        if acc > best_acc:
            best_acc = acc
            best_ratio = sorted(ratios)

    print('Best acc:', best_acc, 'ratio:', best_ratio)

if __name__=='main':
    xmlfile = 'DATA/tt.xml'
    obj_info = ['aaa.png', 'roads',1, 2, 3, 4]
    append_obj_in_xml(xmlfile, obj_info)