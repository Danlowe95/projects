"""

TODO:
DONE 1.) Restrict the polygon to rigid sides and 4 points (should not take too long)
2.) add another polygon (this might take some time) - you can use a button click or a click down/up not near a current point to add a polygon
done 3.) Input methods
  done  3a.) click down, drag, click up [input method]
  done  3b.) click down, click up, drag, click down, clock up [input method]

Interactive tool to draw mask on an image or image-like array.

Adapted from matplotlib/examples/event_handling/poly_editor.py
Jan 9 2014: taken from: https://gist.github.com/tonysyu/3090704
"""
from __future__ import division, print_function
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
    def __init__(self, ax, poly_xy=None, max_ds=10, line_width=2,
                 line_color=(0, 0, 1), face_color=(1, .5, 0)):
        self.showverts = True
        self.max_ds = max_ds
        if poly_xy is None:
            poly_xy = default_vertices(ax)
            
        self.poly = Polygon(poly_xy, animated=True,
                            fc=face_color, ec='none', alpha=0.4)

        ax.add_patch(self.poly)
        ax.set_clip_on(False)
        ax.set_title("Click and drag a point to move it; or click once, then click again."
                     "\n"
                     "Close figure when done.")
        self.ax = ax

        x, y = zip(*self.poly.xy)
        #line_color = 'none'
        color = np.array(line_color) * .6
        marker_face_color = line_color
        line_kwargs = {'lw': line_width, 'color': color, 'mfc': marker_face_color}
        self.line = plt.Line2D(x, y, marker='o', alpha=0.8, animated=True, **line_kwargs)
        self._update_line()
        self.ax.add_line(self.line)

        self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas = self.poly.figure.canvas
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        #canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    def get_mask(self, shape):
        """Return image mask given by mask creator"""
        mask = verts_to_mask(shape, self.verts)
        return mask

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        #Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def draw_callback(self, event):
        #print('[mask] draw_callback(event=%r)' % event)
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self._ind is None:
            self._ind = None;
            return
        ignore = not self.showverts or event.inaxes is None or event.button != 1
        if ignore:
            return
        self._ind = self.get_ind_under_cursor(event)
        if self._ind != None:
            self.indX, self.indY = self.poly.xy[self._ind]
        self.mouseX,self.mouseY = event.xdata, event.ydata

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        ignore = not self.showverts or event.button != 1
        if ignore:
            return
        if self._ind is None:
            return
        currX, currY = self.poly.xy[self._ind]
        #print (currX, ' ', currY)
        #print (math.fabs(self.indX - currX)<3, ' and ', math.fabs(self.indY-currY)<3)
        if math.fabs(self.indX - currX)<3  and math.fabs(self.indY-currY)<3:
            return
        self._ind = None

    
    # def key_press_callback(self, event):
    #     'whenever a key is pressed'
    #     if not event.inaxes:
    #         return
    #     if event.key == 't':
    #         self.showverts = not self.showverts
    #         self.line.set_visible(self.showverts)
    #         if not self.showverts:
    #             self._ind = None
    #     elif event.key == 'd':
    #         ind = self.get_ind_under_cursor(event)
    #         if ind is None:
    #             return
    #         if ind == 0 or ind == self.last_vert_ind:
    #             print('[mask] Cannot delete root node')
    #             return
    #         self.poly.xy = [tup for i, tup in enumerate(self.poly.xy) if i != ind]
    #         self._update_line()
    #     elif event.key == 'i':
    #         xys = self.poly.get_transform().transform(self.poly.xy)
    #         p = event.x, event.y  # cursor coords
    #         for i in range(len(xys) - 1):
    #             s0 = xys[i]
    #             s1 = xys[i + 1]
    #             d = dist_point_to_segment(p, s0, s1)
    #             if d <= self.max_ds:
    #                 self.poly.xy = np.array(
    #                     list(self.poly.xy[:i + 1]) +
    #                     [(event.xdata, event.ydata)] +
    #                     list(self.poly.xy[i + 1:]))
    #                 self._update_line()
    #                 break
    #     self.canvas.draw()
    

    def motion_notify_callback(self, event):
        'on mouse movement'
        # ignore = (not self.showverts or event.inaxes is None or
        #           event.button != 1 or self._ind is None)
        ignore = (not self.showverts or event.inaxes is None)
        if ignore:
            #print ('verts ', not self.showverts, ' inaxes ', event.inaxes, ' event.buton ' ,event.button !=1, ' ind ', self._ind )
            return

        if self._ind is None and event.button ==1:
            'move all vertices'
            self.move_rectangle(event)



            self._update_line()
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.poly)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
            self._ind = None
            'set new mouse loc'
            self.mouseX,self.mouseY = event.xdata, event.ydata 





        # if self._ind is None:
        #     #create new poly
        #     poly2_xy = vertices_under_cursor(event)
        #     self.poly2 = Polygon(poly2_xy, animated=True,
        #                     fc=face_color, ec='none', alpha=0.4)

        # #ax.add_patch(self.poly2)

        #     #Grab  3rd(?) ind

        if self._ind is None:
            return
        self.calculate_move(event)


        self._update_line()

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def move_rectangle(self,event):
        selectedX, selectedY = (self.poly.xy[1])
        beforeX, beforeY = (self.poly.xy[0])
        afterX, afterY = (self.poly.xy[2])
        acrossX, acrossY = (self.poly.xy[3])
        listX = [selectedX, beforeX, afterX, acrossX]
        listY = [selectedY, beforeY, afterY, acrossY]
        maxX = max(listX)
        maxY = max(listY)
        minX = min(listX)
        minY = min(listY)
        x, y = event.xdata, event.ydata
        if x < minX or x> maxX or y<minY or y>maxY:
            return
        # Change selected
        self.poly.xy[1] = selectedX+(x-self.mouseX), selectedY+(y-self.mouseY)

        # Change before vert
        self.poly.xy[0] = beforeX+(x-self.mouseX), beforeY+(y-self.mouseY)
        self.poly.xy[self.last_vert_ind] = beforeX+(x-self.mouseX), beforeY+(y-self.mouseY)
        
        # Change after vert
        self.poly.xy[2] = afterX+(x-self.mouseX), afterY+(y-self.mouseY)
        
        #Change across vert
        self.poly.xy[3] = acrossX+(x-self.mouseX), acrossY+(y-self.mouseY)



    def calculate_move(self,event):
        indBefore = self._ind-1
        if(indBefore < 0):
            indBefore = len(self.poly.xy)-2
        indAfter = (self._ind+1)%4
        selectedX, selectedY = (self.poly.xy[self._ind])
        beforeX, beforeY = (self.poly.xy[indBefore])
        afterX, afterY = (self.poly.xy[indAfter])
        
        changeBefore = -1
        keepX, changeY = -1, -1
        changeAfter = -1
        changeX, keepY = -1, -1

        if beforeX != selectedX:
            changeBefore = indBefore
            keepX, changeY = self.poly.xy[indBefore]
            changeAfter = indAfter
            changeX, keepY = self.poly.xy[indAfter]
        else:
            changeBefore = indAfter
            keepX, changeY = self.poly.xy[indAfter]
            changeAfter = indBefore
            changeX, keepY = self.poly.xy[indBefore]

        x, y = event.xdata, event.ydata

        # Change selected
        if self._ind == 0 or self._ind == self.last_vert_ind:
            self.poly.xy[0] = x, y
            self.poly.xy[self.last_vert_ind] = x, y
        else:
            self.poly.xy[self._ind] = x, y

        # Change vert
        if changeBefore == 0 or changeBefore == self.last_vert_ind:
            self.poly.xy[0] = keepX, y
            self.poly.xy[self.last_vert_ind] = keepX, y
        else:
            self.poly.xy[changeBefore] = keepX, y

        # Change horiz
        if changeAfter == 0 or changeAfter == self.last_vert_ind:
            self.poly.xy[0] = x, keepY
            self.poly.xy[self.last_vert_ind] = x, keepY
        else:
            self.poly.xy[changeAfter] = x, keepY

    def _update_line(self):
        # save verts because polygon gets deleted when figure is closed
        self.verts = self.poly.xy
        self.last_vert_ind = len(self.poly.xy) - 1
        self.line.set_data(zip(*self.poly.xy))

    def get_ind_under_cursor(self, event):
        'get the index of the vertex under cursor if within max_ds tolerance'
        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
        if d[ind] >= self.max_ds:
            ind = None
        return ind


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
    try:
        from hscom import fileio as io
        img = io.imread('/lena.png', 'RGB')
    except ImportError as ex:
        print('cant read lena: %r' % ex)
        img = np.random.uniform(255, 255, size=(100, 100))

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
