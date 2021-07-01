import xml.etree.ElementTree as ET


def append_obj_in_xml(xmlfile, obj_info):
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

if __name__=='main':
    xmlfile = 'DATA/tt.xml'
    obj_info = ['aaa.png', 'roads',1, 2, 3, 4]
    append_obj_in_xml(xmlfile, obj_info)

