import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image

matplotlib.use('Agg')


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


class PathVisualizer:
    def __init__(self,
                 figsize_per_example=(4, 4),
                 background_image=None,
                 color_cycle=None,
                 cmap=None,
                 marker_cycle=None):

        self.figsize_per_example = figsize_per_example
        self.background_image = background_image
        if cmap is not None and color_cycle is not None:
            raise ValueError('cmap and color cycle cannot be set at the same time')

        self.cmap = cmap or plt.cm.get_cmap('rainbow')
        self.color_cycle = color_cycle
        self.marker_cycle = marker_cycle

    def plot(self, *paths):
        batch_size = len(paths[0])
        color_cycle = self.color_cycle or self.cmap(np.linspace(0, 1, len(paths)))

        figx, figy = self.figsize_per_example
        fig, axes = plt.subplots(nrows=batch_size, ncols=1, figsize=(figx, batch_size * figy))
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

                ax = axes[i] if batch_size > 1 else axes
                if self.background_image is not None:
                    ax.imshow(self.background_image, extent=(-1, 1, 1, -1))

                ax.scatter(path[i][:1, 0], path[i][:1, 1], c=color, marker='x', s=100)
                ax.plot(path[i][1:, 0], path[i][1:, 1], c=color, marker=marker)
                ax.set_xlim((-1, 1))
                ax.set_ylim((-1, 1))
                ax.invert_yaxis()
        return fig
