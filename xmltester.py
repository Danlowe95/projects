from xml.dom import minidom
import xml.etree.ElementTree as et

# xmldoc = minidom.parse('test.xml')
# itemlist = xmldoc.getElementsByTagName('coord')
# if(itemlist.length > 1):
# 	print itemlist
tree = et.parse('test.xml')
root = tree.getroot()
for neighbor in root.iter('coord'):
	print neighbor.get('x')