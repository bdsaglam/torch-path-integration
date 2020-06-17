import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image

matplotlib.use('Agg')


class PathVisualizer:
    def __init__(self,
                 rect=(-1, 1, 1, -1),  # (left, right, bottom, top)
                 figsize_per_example=(4, 4),
                 bg_image=None,
                 color_cycle=None,
                 cmap=None,
                 marker_cycle=None):

        self.rect = rect
        self.figsize_per_example = figsize_per_example
        self.bg_image = bg_image
        if cmap is not None and color_cycle is not None:
            raise ValueError('cmap and color cycle cannot be set at the same time')

        self.cmap = cmap or plt.cm.get_cmap('rainbow')
        self.color_cycle = color_cycle
        self.marker_cycle = marker_cycle

    def plot(self, *paths):
        batch_size = len(paths[0])
        color_cycle = self.color_cycle or self.cmap(np.linspace(0, 1, len(paths)))

        figx, figy = self.figsize_per_example
        fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=(2 * figx, batch_size * figy))
        for i in range(batch_size):
            for j in range(len(paths)):
                path = paths[j]
                color = color_cycle[j % len(color_cycle)]
                if isinstance(color, np.ndarray):
                    color = color[None, :]

                if self.marker_cycle is not None:
                    marker = self.marker_cycle[j % len(self.marker_cycle)]
                else:
                    marker = None

                axl, axr = axes[i] if batch_size > 1 else axes

                # paint bg image to left subplot
                if self.bg_image is not None:
                    axl.imshow(self.bg_image, extent=self.rect)

                # draw path on both left and right subplots
                for ax in (axl, axr):
                    ax.scatter(path[i][:1, 0], path[i][:1, 1], c=color, marker='x', s=100)
                    ax.plot(path[i][:, 0], path[i][:, 1], c=color, marker=marker)

                # set rect limits on left subplot
                axl.set_xlim((self.rect[0], self.rect[1]))
                axl.set_ylim((self.rect[3], self.rect[2]))

                # make both subplots bottom-top
                axl.invert_yaxis()
                axr.invert_yaxis()

        return fig


def canvas_to_image(canvas):
    canvas.draw()
    return Image.frombytes('RGB',
                           canvas.get_width_height(),
                           canvas.tostring_rgb())


def fig_to_tensor(fig):
    img = canvas_to_image(fig.canvas)
    return torchvision.transforms.ToTensor()(img)


def plot_location_predictions(initial_location, prediction, target):
    batch_size = prediction.shape[0]
    fig, axes = plt.subplots(nrows=batch_size, ncols=1, figsize=(4, batch_size * 4))
    for i in range(batch_size):
        ax = axes[i] if batch_size > 1 else axes
        ax.scatter(initial_location[i, :, 0], initial_location[i, :, 1], c='black', marker='x')
        ax.plot(target[i, :, 0], target[i, :, 1], c='blue', marker='.')
        ax.plot(prediction[i, :, 0], prediction[i, :, 1], c='red', marker='.')
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.invert_yaxis()
    return fig
