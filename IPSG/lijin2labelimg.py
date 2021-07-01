from pathlib import Path
from xml.dom.minidom import parse
import xml.dom.minidom
import xml.etree.ElementTree as ET


def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


for xml_file in Path('D:/标注1295目标检测/li_jin/outputs').glob('*.xml'):
    parser = ET.parse(str(xml_file))
    xml_root = parser.getroot()

    # print(xml_root.tag, ":", xml_root.attrib)  # 打印根元素的tag和属性

    outputs = xml_root.find('outputs')

    # print(outputs.tag, ":", outputs.text)
    # print(len(outputs))
    if len(outputs) == 0:
        continue

    root_ = ET.Element('annotation')
    filename_ = ET.SubElement(root_, 'filename')
    filename_.text = xml_file.name[:-3] + 'jpg'

    for item in outputs.find('object').findall('item'):
        name = item.find('name').text
        xmin = item.find('bndbox').find('xmin').text
        ymin = item.find('bndbox').find('ymin').text
        xmax = item.find('bndbox').find('xmax').text
        ymax = item.find('bndbox').find('ymax').text

        obj_ = ET.SubElement(root_, 'object')
        name_ = ET.SubElement(obj_, 'name')
        bndbox_ = ET.SubElement(obj_, 'bndbox')
        xmin_ = ET.SubElement(bndbox_, 'xmin')
        ymin_ = ET.SubElement(bndbox_, 'ymin')
        xmax_ = ET.SubElement(bndbox_, 'xmax')
        ymax_ = ET.SubElement(bndbox_, 'ymax')
        name_.text = name
        xmin_.text = xmin
        ymin_.text = ymin
        xmax_.text = xmax
        ymax_.text = ymax

    tree_ = ET.ElementTree(root_)
    # 在终端显示整个xml内容
    ET.dump(root_)
    # 写入xml文件
    pretty_xml(root_, '\t', '\n')  # 执行美化方法
    tree_.write(xml_file.name, encoding="utf-8", xml_declaration=True, method='xml')

    # # 添加子节点
    #
    # # 添加text，即22，字符串格式
    #
    # gender1 = ET.SubElement(person1, 'gender')
    # gender1.text = 'male'
    # person2 = ET.SubElement(root, 'person', {'name': 'Yellow'})
    # age2 = ET.SubElement(person2, 'age')
    # age2.text = '20'
    # gender2 = ET.SubElement(person2, 'gender')
    # gender2.text = 'female'
    # # 将根目录转化为xml树状结构(即ElementTree对象)
