import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
IMG_SIZE = 28

class DrawDigits:
    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size
        self.drawing = False
        self.image = np.zeros((img_size, img_size))
        self.fig, self.ax = plt.subplots()
        self.canvas = self.ax.imshow(self.image, cmap='gray_r')
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.set_title('Draw a digit')
        self.ax.axis('off')
        plt.show()

    def on_press(self, event):
        self.drawing = True
        self.update_image(event)

    def on_move(self, event):
        if self.drawing:
            self.update_image(event)

    def on_release(self, event):
        self.drawing = False
        self.get_image()

    def update_image(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        self.image[y-1:y+2, x-1:x+2] = 1
        self.canvas.set_data(self.image)
        self.fig.canvas.draw()

    def get_image(self):
        return self.image

# Draw a digit
drawer = DrawDigits()
plt.show()