#jonathan wrona
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

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
#	Author: Jonathan Wrona
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
		canvas.mpl_connect('motion_notify_event', self.on_move)

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

	def button_release_callback(self, event):
		self.button_pressed = False
		previous_pos_x = -1.0
		previous_pos_y = -1.0
		if(len(self.curr_event.findall('coord')) <= 0):
			self.root.remove(self.root[-1])
		#UNCOMMENT IF UPDATING AFTER EACH EVENT$
		#tree = et.ElementTree(self.root)
		#tree.write(self.file_)

	def on_move(self, event):
		if(self.button_pressed):
			current_pos_x = int(math.floor(event.xdata))
			current_pos_y = int(math.floor(event.ydata))
			if(not(current_pos_x == self.previous_pos_x and current_pos_y == self.previous_pos_y)):
				print 'updating path'
				self.previous_pos_x = current_pos_x
				self.previous_pos_y = current_pos_y
				coord = et.SubElement(self.curr_event, 'coord')
				coord.set('x', str(current_pos_x))
				coord.set('y', str(current_pos_y))
				#UNCOMMENT IF UPDATING LIVE#
				tree = et.ElementTree(self.root)
				tree.write(self.file_)

def get_mouse_button(button):
	if(button == 1):
		return "left"
	elif(button == 3):
		return "right"

def click_demo():
	f = open('test.xml', 'w')
	f.close()

	img = np.random.uniform(0, 255, size=(100, 100))

	ax = plt.subplot(111)
	ax.imshow(img)

	ct = ClickTracker(ax, 'test.xml')
	plt.title('Click on the image and the corresponding xml will be generated')
	plt.show()


if __name__ == '__main__':
	click_demo()

