from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Loader(object):

    def __init__(self, image_name, size):
        self.image_name = image_name
        self.size = size

    def image_loader(self):
        image = Image.open(self.image_name)
        width, height = image.size
        if width != height:
            difference = width - height
            if difference > 0:
                image = image.crop((difference / 2, 0, height + (difference / 2), height))
            else:
                image = image.crop((0, (-difference) / 2, width, width + ((-difference) / 2)))

        loader = transforms.Compose([
            transforms.Resize(self.size),  # scale imported image
            transforms.ToTensor()])  # transform it into a torch tensor

        image = Variable(loader(image))
        # fake batch dimension required to fit network's input dimensions
        image = image.unsqueeze(0)
        return image


def imshow(tensor, size, final=False, out=None):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, size, size)  # remove the fake batch dimension
    image = unloader(image)
    if final:
        image.save(out)

    # fig = plt.imshow(image)
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    # # if title is not None:
    # #     plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated
