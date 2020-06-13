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
