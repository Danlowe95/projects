from __future__ import division, print_function
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as lines


ax = plt.subplot(111)
circle1=plt.Circle((0.5,.5),.5,color='r')
line_kwargs = {'lw': 2, 'color': 'red', 'mfc': 'b'}
line1 = [(.4, 1), (1, 3)]
x,y = zip(*line1)
ax.add_line(plt.Line2D(x, y, marker='o', alpha=0.8, animated=True, **line_kwargs));
ax.add_patch(circle1)
plt.plot()

plt.show()