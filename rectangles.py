"""

TODO:
1. create a function call to create a rectangle at a given set of points
2.. Make 45 degree rotation - need help


Interactive tool to draw mask on an image or image-like array.

Adapted from matplotlib/examples/event_handling/poly_editor.py
Jan 9 2014: taken from: https://gist.github.com/tonysyu/3090704
"""
from __future__ import division, print_function
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.image as mpimg
from matplotlib.mlab import dist_point_to_segment
import math as math
#from matplotlib import nxutils  # Depricated

# Scientific
import numpy as np


def _nxutils_points_inside_poly(points, verts):
    'nxutils is depricated'
    path = matplotlib.path.Path(verts)
    return path.contains_points(points)


def verts_to_mask(shape, verts):
    print(verts)
    h, w = shape[0:2]
    y, x = np.mgrid[:h, :w]
    points = np.transpose((x.ravel(), y.ravel()))
    #mask = nxutils.points_inside_poly(points, verts)
    mask = _nxutils_points_inside_poly(points, verts)
    return mask.reshape(h, w)


class MaskCreator(object):
    """An interactive polygon editor.

    Parameters
    ----------
    poly_xy : list of (float, float)
        List of (x, y) coordinates used as vertices of the polygon.
    max_ds : float
        Max pixel distance to count as a vertex hit.

    Key-bindings
    ------------
    't' : toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them
    'd' : delete the vertex under point
    'i' : insert a vertex at point.  You must be within max_ds of the
          line connecting two existing vertices
    """
    def __init__(self, ax, poly_xy=None, max_ds=10, line_width=4,      
                 line_color=(1, 1, 1), face_color=(0, 0,0)):
        self.showverts = True
        self.max_ds = max_ds
        self.fc_default = face_color
        self.mouseX = None
        self.mouseY = None
        self._polyHeld = False
        self._thisPoly = None
        self.press1 = False;
        self.canUncolor = False;
        self.poly_count = 0;
        if poly_xy is None:
            poly_xy = default_vertices(ax)
            
            #*********Start of list implementation
            Poly1 = Polygon(poly_xy, animated=True,
                            fc=face_color, ec='none', alpha=0, picker=True)
            self.polyList = [Poly1]
            self.poly_count+=1;
            self.polyList[0].num = 0;

        ax.add_patch(self.polyList[0])
        ax.set_clip_on(False)
        ax.set_title("Click and drag a point to move it; or click once, then click again."
                     "\n"
                     "Close figure when done.")
        self.ax = ax

        x, y = zip(*self.polyList[0].xy)
        #line_color = 'none'
        color = np.array(line_color)
        marker_face_color = line_color
        line_kwargs = {'lw': line_width, 'color': color, 'mfc': marker_face_color}
        self.line = [plt.Line2D(x, y, marker='o', alpha=1, animated=True, **line_kwargs)]
        self._update_line()
        self.ax.add_line(self.line[0])

        self.polyList[0].add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas = self.polyList[0].figure.canvas
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        canvas.mpl_connect('pick_event', self.onpick)
        # canvas.mpl_connect('figure_enter_event', self.mouse_enter)
        # canvas.mpl_connect('figure_leave_event', self.mouse_leave)
        

        self.canvas = canvas

        #second polygon

    
    def rotate45(self, poly):
        'starting to figure out rotation'
        'called when a certain button is clicked (currently when a key is clicked)'

        'move the points clockwise'





    def get_mask(self, shape):
        """Return image mask given by mask creator"""
        mask = verts_to_mask(shape, self.verts)
        return mask

    def update_UI(self):
        self._update_line()
        self.canvas.restore_region(self.background)
        for n, poly in enumerate(self.polyList):
            self.ax.draw_artist(poly)
            self.ax.draw_artist(self.line[n])
        self.canvas.blit(self.ax.bbox)
    
    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line[poly.num].get_visible()
        #Artist.update_from(self.line, poly)
        self.line[poly.num].set_visible(vis)  # don't use the poly visibility state

    def draw_callback(self, event):
        #print('[mask] draw_callback(event=%r)' % event)
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        for n, poly in enumerate(self.polyList):
            self.ax.draw_artist(poly)
            self.ax.draw_artist(self.line[n])
        self.canvas.blit(self.ax.bbox)

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self._ind is None:
            self._ind = None;
            return
        ignore = not self.showverts or event.inaxes is None or event.button != 1
        if ignore:
            return
        # if(self.polyList[1].contains()):
        #     print("true")
        # else:
        #     print("false")
        if self._thisPoly is None:
            print("WARNING: Polygon unknown. Using default.")
            self._thisPoly = self.polyList[0]
        polyind, self._ind = self.get_ind_under_cursor(event)
        if self._ind != None and polyind != None:
            self._thisPoly = self.polyList[polyind]
            self.indX, self.indY = self._thisPoly.xy[self._ind]
            self._polyHeld = True

        self.mouseX,self.mouseY = event.xdata, event.ydata
        if self._polyHeld is True or self._ind is not None:
            self._thisPoly.set_alpha(.2)
            self._thisPoly.set_facecolor('white')
        self.press1 = True;
        self.canUncolor = False;
        self._update_line()
        self.canvas.restore_region(self.background)
        for n, poly in enumerate(self.polyList):
            self.ax.draw_artist(poly)
            self.ax.draw_artist(self.line[n])
        self.canvas.blit(self.ax.bbox)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if self._polyHeld is True and (self._ind == None or self.press1 is False):
            self._polyHeld = False
        
        ignore = not self.showverts or event.button != 1
        if ignore:
            return
        #print("self.uncolor: ", self.canUncolor, " self.press1: ", self.press1, " self._thisPoly: ", self._thisPoly)
        

        


        if (self._ind is None) or self._polyHeld is False or (self._ind is not None and self.press1 is True) and self._thisPoly is not None and self.canUncolor is True:
            self._thisPoly.set_alpha(0)

        # if self.canUncolor is True and self.press1 is True:
        #     print("trues")
        #     self._thisPoly.set_facecolor(self.fc_default)
        self.update_UI();
        self.press1 = False;

        if self._ind is None:
            return
        if self._thisPoly is None:
            print("WARNING: Polygon unknown. Using default. (2)")
            self._thisPoly = self.polyList[0]
        currX, currY = self._thisPoly.xy[self._ind]

        

        if math.fabs(self.indX - currX)<3  and math.fabs(self.indY-currY)<3:
            
            return
        if (self._ind is None) or self._polyHeld is False or (self._ind is not None and self.press1 is True) and self._thisPoly is not None:
            self._thisPoly = None
        self._ind = None
        self._polyHeld = False;

    def draw_new_poly(self):
        coords = default_vertices(self.ax)
        Poly = Polygon(coords, animated=True,
                            fc="white", ec='none', alpha=0.2, picker=True)
        self.polyList.append(Poly)
        self.ax.add_patch(Poly)
        x, y = zip(*Poly.xy)
        #line_color = 'none'
        color = np.array((1,1,1))
        marker_face_color = (1,1,1)
        line_width = 4;
        
        line_kwargs = {'lw': line_width, 'color': color, 'mfc': marker_face_color}
        self.line.append(plt.Line2D(x, y, marker='o', alpha=1, animated=True, **line_kwargs))
        self._update_line()
        self.ax.add_line(self.line[-1])

        Poly.add_callback(self.poly_changed)
        Poly.num = self.poly_count
        self._ind = None  # the active vert
        self.poly_count+=1;


    
    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        if event.key == 't':
            self.draw_new_poly();
        # elif event.key == 'd':
        #     ind = self.get_ind_under_cursor(event)
        #     if ind is None:
        #         return
        #     if ind == 0 or ind == self.last_vert_ind:
        #         print('[mask] Cannot delete root node')
        #         return
        #     self.poly.xy = [tup for i, tup in enumerate(self.poly.xy) if i != ind]
        #     self._update_line()
        # elif event.key == 'i':
        #     xys = self.poly.get_transform().transform(self.poly.xy)
        #     p = event.x, event.y  # cursor coords
        #     for i in range(len(xys) - 1):
        #         s0 = xys[i]
        #         s1 = xys[i + 1]
        #         d = dist_point_to_segment(p, s0, s1)
        #         if d <= self.max_ds:
        #             self.poly.xy = np.array(
        #                 list(self.poly.xy[:i + 1]) +
        #                 [(event.xdata, event.ydata)] +
        #                 list(self.poly.xy[i + 1:]))
        #             self._update_line()
        #             break
        self.canvas.draw()
    

    def motion_notify_callback(self, event):
        'on mouse movement'
        # ignore = (not self.showverts or event.inaxes is None or
        #           event.button != 1 or self._ind is None)
        ignore = (not self.showverts or event.inaxes is None)
        if ignore:
            #print ('verts ', not self.showverts, ' inaxes ', event.inaxes, ' event.buton ' ,event.button !=1, ' ind ', self._ind )
            return
        if self.press1 is True:
            self.canUncolor = True;
        if self._ind is None and event.button ==1:
            'move all vertices'
            if self._polyHeld is True:
                self.move_rectangle(event, self._thisPoly, event.xdata, event.ydata)


            self.update_UI();
            self._ind = None
            'set new mouse loc'
            self.mouseX,self.mouseY = event.xdata, event.ydata 

        # if event.button is None and self._ind is None:
        #     print("eher")
        #     for poly in self.polyList:
        #         if poly.contains(event):
        #             print("hereee")
        #             poly.set_alpha(.9)
        #         else:
        #             ply.set_alpha(0)

        #     self.update_UI();   
        if self._ind is None:
            return
        if self._polyHeld is True:
            self.calculate_move(event, self._thisPoly)
        else:
            print("error no poly known")

        self.update_UI();
    def onpick(self, event):
        print("onpick")
        self._thisPoly = event.artist
        x,y = event.mouseevent.xdata, event.mouseevent.xdata
        #self.move_rectangle(event, thisPoly, x, y)
        self._polyHeld = True;


    def mouse_enter(self, event):
        print(event.name)
        # if(event.name != 'Polygon'):
        #     return
        self._thisPoly = event.artist;
        self._thisPoly.set_alpha(.2)


    def mouse_leave(self, event):
        self._thisPoly.set_alpha(0)
        self._thisPoly = None;



    def check_dims(self, coords):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        # print("coords 0: ", coords[0], "coords1: ", coords[1], "xlim: ", xlim, "ylim: ", ylim)
        if coords[0] >= xlim[0] and coords[0] <= xlim[1] and coords[1] >= ylim[1] and coords[1] <= ylim[0]:
            return True
        return False

    def move_rectangle(self,event, polygon, x, y):
        selectedX, selectedY = (polygon.xy[1])
        beforeX, beforeY = (polygon.xy[0])
        afterX, afterY = (polygon.xy[2])
        acrossX, acrossY = (polygon.xy[3])
        'if we are not holding a rectangle, return'
        if self._polyHeld is not True:
            return
        # Change selected

        new1 = selectedX + (x - self.mouseX), selectedY + (y - self.mouseY)
        new0 = beforeX + (x - self.mouseX), beforeY + (y - self.mouseY)
        new2 = afterX + (x - self.mouseX), afterY + (y - self.mouseY)
        new3 = acrossX + (x - self.mouseX), acrossY + (y - self.mouseY)
        if(self.check_dims(new1)) is True and self.check_dims(new0) is True and self.check_dims(new2) is True and self.check_dims(new3) is True:
            polygon.xy[1] = new1

            # Change before vert
            polygon.xy[0] = new0
            polygon.xy[self.last_vert_ind] = new0
            # Change after vert
            polygon.xy[2] = new2
            
            #Change across vert
            polygon.xy[3] = new3



    def calculate_move(self,event, poly):
        indBefore = self._ind-1
        if(indBefore < 0):
            indBefore = len(poly.xy)-2
        indAfter = (self._ind+1)%4
        selectedX, selectedY = (poly.xy[self._ind])
        beforeX, beforeY = (poly.xy[indBefore])
        afterX, afterY = (poly.xy[indAfter])
        
        changeBefore = -1
        keepX, changeY = -1, -1
        changeAfter = -1
        changeX, keepY = -1, -1

        if beforeX != selectedX:
            changeBefore = indBefore
            keepX, changeY = poly.xy[indBefore]
            changeAfter = indAfter
            changeX, keepY = poly.xy[indAfter]
        else:
            changeBefore = indAfter
            keepX, changeY = poly.xy[indAfter]
            changeAfter = indBefore
            changeX, keepY = poly.xy[indBefore]

        x, y = event.xdata, event.ydata

        # Change selected
        if self._ind == 0 or self._ind == self.last_vert_ind:
            poly.xy[0] = x, y
            poly.xy[self.last_vert_ind] = x, y
        else:
            poly.xy[self._ind] = x, y

        # Change vert
        if changeBefore == 0 or changeBefore == self.last_vert_ind:
            poly.xy[0] = keepX, y
            poly.xy[self.last_vert_ind] = keepX, y
        else:
            poly.xy[changeBefore] = keepX, y

        # Change horiz
        if changeAfter == 0 or changeAfter == self.last_vert_ind:
            poly.xy[0] = x, keepY
            poly.xy[self.last_vert_ind] = x, keepY
        else:
            poly.xy[changeAfter] = x, keepY

    def _update_line(self):
        # save verts because polygon gets deleted when figure is closed
        for n, poly in enumerate(self.polyList):
            self.verts = poly.xy
            self.last_vert_ind = len(poly.xy) - 1
            self.line[n].set_data(zip(*poly.xy))

    def get_ind_under_cursor(self, event):
        'get the index of the vertex under cursor if within max_ds tolerance'
        # display coords
        'need to get the list of all inds of all polygon'
        #what I planned to do:
        #for x in range (0, len(self.polyList)-1):
            #xy += np.asarray(self.polyList[0].xy)



        def get_ind_and_dist(poly):
            xy = np.asarray(poly.xy)
            xyt = poly.get_transform().transform(xy)
            xt, yt = xyt[:, 0], xyt[:, 1]
            d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
            indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
            ind = indseq[0]
            mindist = d[ind]
            if mindist >= self.max_ds:
                ind = None
                mindist = None
            return (ind, mindist)
        ind_dist_list = [get_ind_and_dist(poly) for poly in self.polyList]
        min_dist = None
        min_ind  = None
        sel_polyind = None
        for polyind, (ind, dist) in enumerate(ind_dist_list):
            if ind is None:
                continue
            if min_dist is None:
                min_dist = dist
                min_ind = ind
                sel_polyind = polyind
            elif dist < min_dist:
                min_dist = dist
                min_ind = ind
                sel_polyind = polyind
        return (sel_polyind, min_ind)


def default_vertices(ax):
    """Default to rectangle that has a quarter-width/height border."""
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    w = np.diff(xlims)
    h = np.diff(ylims)
    x1, x2 = xlims + w // 4 * np.array([1, -1])
    y1, y2 = ylims + h // 4 * np.array([1, -1])
    return ((x1, y1), (x1, y2), (x2, y2), (x2, y1))


def vertices_under_cursor(event):
    """Create 5 ind on one point"""
    x1 = event.xdata
    y1 = event.ydata
    return ((x1, y1), (x1, y1), (x1, y1,), (x1, y1))

def apply_mask(img, mask):
    masked_img = img.copy()
    masked_img[~mask] = np.uint8(np.clip(masked_img[~mask] - 100., 0, 255))
    return masked_img


def roi_to_verts(roi):
    (x, y, w, h) = roi
    verts = np.array([(x + 0, y + h),
                      (x + 0, y + 0),
                      (x + w, y + 0),
                      (x + w, y + h),
                      (x + 0, y + h)], dtype=np.float32)
    return verts


def roi_to_mask(shape, roi):
    verts = roi_to_verts(roi)
    mask = verts_to_mask(shape, verts)
    return mask


def mask_creator_demo(mode=0):
    print('*** START DEMO ***')
    print('mode = %r' % mode)
    # try:
    #     from hscom import fileio as io
    #     img = io.imread('/zebra.jpg', 'RGB')
    # except ImportError as ex:
    #     print('cant read zebra: %r' % ex)
    #     img = np.random.uniform(255, 255, size=(100, 100))

    img = mpimg.imread('zebra2.jpg')
    ax = plt.subplot(111)
    ax.imshow(img)

    if mode == 0:
        mc = MaskCreator(ax)
        # Do interaction
        plt.show()
        # Make mask from selection
        mask = mc.get_mask(img.shape)
        # User must close previous figure
    elif mode == 1:
        from hsgui import guitools
        from hsviz import draw_func2 as df2
        ax.set_title('Click two points to select an ROI (old mode)')
        # Do selection
        roi = guitools.select_roi()
        # Make mask from selection
        mask = roi_to_mask(img.shape, roi)
        # Close previous figure
        df2.close_all_figures()

    # Modify the image with the mask
    masked_img = apply_mask(img, mask)
    # show the modified image
    plt.imshow(masked_img)
    plt.title('Region outside of mask is darkened')
    print('show2')
    plt.show()


if __name__ == '__main__':
    import sys
    print(sys.argv)
    if len(sys.argv) == 1:
        mode = 0
    else:
        mode = 1
    mask_creator_demo(mode)
