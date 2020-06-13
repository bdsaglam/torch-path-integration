import numpy as np
import torchvision
from PIL import Image


def canvas_to_image(canvas):
    canvas.draw()
    return Image.frombytes('RGB',
                           canvas.get_width_height(),
                           canvas.tostring_rgb())


def fig_to_tensor(fig):
    img = canvas_to_image(fig.canvas)
    return torchvision.transforms.ToTensor()(img)


def truncated_normal_(tensor, mean=0, std=1):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def init_trunc_normal(t, size):
    std = 1. / np.sqrt(size)
    return truncated_normal_(t, 0, std)
