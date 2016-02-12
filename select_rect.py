import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class SelectRect:
    '''
    Select area of image data by dragging mouse to create one rectangular
    region
    '''
    def __init__(self, ax):
        self.ax = ax
        self.done = False
        self.press = None
        self.title = ax.get_title()
        self.x = None
        self.y = None
        ax.set_title('Drag to select area.')
        connect = self.ax.figure.canvas.mpl_connect
        self.cidpress = connect('button_press_event', self.on_press)
        self.cidmotion = connect('motion_notify_event', self.on_motion)
        self.cidrelease = connect('button_release_event', self.on_release)

    def on_press(self, event):
        if event.button == 1:
            # Left click
            self.press = event.xdata, event.ydata
            rect = Rectangle(self.press, 1, 1, alpha=.1, color='lime')
            self.ax.add_patch(rect)
            self.rect = rect
            plt.pause(.1)
        elif event.button == 3:
            # Right click
            self.disconnect()

    def on_motion(self, event):
        if self.press is None: return
        x0, y0 = self.press
        x1, y1 = event.xdata, event.ydata
        if not any([p is None for p in (x0, x1, y0, y1)]):
            self.rect.set_width(x1 - x0)
            self.rect.set_height(y1 - y0)
        self.ax.figure.canvas.draw()
        plt.pause(.1)

    def on_release(self, event):
        self.ax.figure.canvas.draw()
        x0, y0 = self.press
        x1, y1 = event.xdata, event.ydata
        xl, xr = int(min(x0, x1)), int(max(x0, x1))
        yt, yb = int(min(y0, y1)), int(max(y0, y1))
        self.x = (xl, xr)
        self.y = (yt, yb)
        self.disconnect()
        plt.pause(.1)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.ax.set_title(self.title)
        self.ax.figure.canvas.draw()
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)
