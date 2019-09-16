# replace info in xml files
# here is an example for change all object/name in the xml files
import os
import os.path
from xml.etree import ElementTree as et
from xml.dom import minidom
from xml.etree import ElementTree
# path to the xml files folder
path = "/Users/mac/Desktop/spider/dataset/testsxml/"
files = os.listdir(path)  
s = []
count = 0
for xmlFile in files: 
    if xmlFile.endswith('.xml'):
        tree = et.parse(path+xmlFile)
        for i in tree.findall('object/name'):
            i.text = 'spider'
            tree.write(path+xmlFile)