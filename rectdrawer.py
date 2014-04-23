#dan lowe/ jonathan wrona
from xml.dom import minidom
from scipy import misc
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2

##############################################################
#
#	ClickTracker2.py is a snippet of code for tracking the
#	clicks that a user has made on an image and writing that
#	data to an xml file. To test that xml file is named
#	test.xml in the same folder as ClickTracker.py.
#
#	This version of ClickTracker can update the file after
#	each event or live.
#
#	Author: Dan Lowe/Jonathan Wrona
#
##############################################################

class ClickTracker(object):

	file_ = 'test.xml'

	previous_pos_x = -1.0
	previous_pos_y = -1.0

	button_pressed = False

	root = et.Element('data')
	curr_event = et.Element('event')

	brush_size = 10

	def __init__(self, ax, file_):
		self.showverts = True
		self.figure = plt.figure(1)

		self.file_ = file_

		canvas = self.figure.canvas
		canvas.mpl_connect('button_press_event', self.button_press_callback)
		canvas.mpl_connect('button_release_event', self.button_release_callback)
		#canvas.mpl_connect('motion_notify_event', self.on_move)

	def button_press_callback(self, event):
		if(event.button == 1 or event.button == 3):
			self.button_pressed = True
			self.curr_event = et.SubElement(self.root, 'event')
			self.curr_event.set('button', get_mouse_button(event.button))
			self.curr_event.set('size', str(self.brush_size))
			current_pos_x = int(math.floor(event.xdata))
			current_pos_y = int(math.floor(event.ydata))
			coord = et.SubElement(self.curr_event, 'coord')
			coord.set('x', str(current_pos_x))
			coord.set('y', str(current_pos_y))
			#UNCOMMENT IF UPDATING LIVE#
			tree = et.ElementTree(self.root)
			tree.write(self.file_)
			has_two_points()

	def button_release_callback(self, event):
		self.button_pressed = False
		previous_pos_x = -1.0
		previous_pos_y = -1.0
		if(len(self.curr_event.findall('coord')) <= 0):
			self.root.remove(self.root[-1])
		#UNCOMMENT IF UPDATING AFTER EACH EVENT$
		#tree = et.ElementTree(self.root)
		#tree.write(self.file_)

	# def on_move(self, event):
	# 	if(self.button_pressed):
	# 		current_pos_x = int(math.floor(event.xdata))
	# 		current_pos_y = int(math.floor(event.ydata))
	# 		if(not(current_pos_x == self.previous_pos_x and current_pos_y == self.previous_pos_y)):
	# 			print 'updating path'
	# 			self.previous_pos_x = current_pos_x
	# 			self.previous_pos_y = current_pos_y
	# 			coord = et.SubElement(self.curr_event, 'coord')
	# 			coord.set('x', str(current_pos_x))
	# 			coord.set('y', str(current_pos_y))
	# 			#UNCOMMENT IF UPDATING LIVE#
	# 			tree = et.ElementTree(self.root)
	# 			tree.write(self.file_)
	# 			has_two_points()

def get_mouse_button(button):
	if(button == 1):
		return "left"
	elif(button == 3):
		return "right"

def has_two_points():
	# xmldoc = minidom.parse('test.xml')
	# itemlist = xmldoc.getElementsByTagName('event')
	# if(itemlist.length > 1):
	tree = et.parse('test.xml')
	root = tree.getroot()
	print len(list(root))
	if(len(list(root)) %2 == 0):
		pts = []

		for neighbor in root.iter('coord'):
			x = int(neighbor.get('x'))
			y = int(neighbor.get('y'))
			pts.append((x,y))
		print 'img draw'
		cv2.rectangle(img, pts[len(pts)-2], pts[len(pts)-1], (255), 8)
		ax.images.pop() 
		ax.imshow(img) 
		plt.draw()

def click_demo():
	f = open('test.xml', 'w')
	f.close()
	print 'img assign'
	global img 
	#img = misc.lena()
	origin = ''
	img = mpimg.imread('zebra.jpg')
	#img = np.random.uniform(255, 0, size=(100, 100))

	global ax 
	ax = plt.subplot(111)
	ax.imshow(img)

	ct = ClickTracker(ax, 'test.xml')
	plt.title('Choose two points for your rectange by left clicking')
	plt.show()
	
	
	# itemlist = xmldoc.getElementsByTagName('event')
	# print len(itemList)
	# print itemlist[0].attributes['left'].value
	# for s in itemlist :
	# 	print s.attributes['name'].value
####create rectangle
if __name__ == '__main__':
	click_demo()

