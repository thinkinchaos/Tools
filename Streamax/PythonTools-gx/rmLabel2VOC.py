# -*- coding:utf-8 -*-
import os
import os.path


import  xml.dom.minidom

dom = xml.dom.minidom.parse('000131.xml')

root = dom.documentElement
print(root.nodeName)
print(root.nodeValue)
print(root.nodeType)
print(root.ELEMENT_NODE)
print('Done!')

object = dom.object
print(object.name)