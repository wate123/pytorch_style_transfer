from __future__ import print_function
import matplotlib.pyplot as plt
import torchvision.models as models

from net import *
import utils as ul
import style

from option import Options
import os
import time


def main():
    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    # figure out the experiments type
    args = Options().parse()
    if args.subcommand is None:
        raise ValueError("ERROR: specify the experiment type")
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    if args.subcommand == 'optim':

        # desired size of the output image
        # imsize = 256 if use_cuda else 128  # use small size if no gpu
        args = Options().parse()

        style_img = ul.Loader(args.style_image, args.style_size).image_loader().type(dtype)
        content_img = ul.Loader(args.content_image, args.content_size).image_loader().type(dtype)

        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"

        plt.ion()
        # Display the Input Images
        plt.figure()
        ul.imshow(content_img.data, args.content_size)

        plt.figure()
        ul.imshow(style_img.data, args.content_size)

        cnn = models.vgg19(pretrained=True).features
        # move it to the GPU if possible:
        if use_cuda:
            cnn = cnn.cuda()

        input_img = content_img.clone()

        # if you want to use a white noise instead uncomment the below line:
        # input_img = Variable(torch.randn(content_img.data.size())).type(dtype)

        # Finally, run the algorithm

        output = style.run_style_transfer(cnn, content_img, style_img, input_img,
                                    num_steps=args.iters, style_weight=args.style_weight,
                                    content_weight=args.content_weight)

        fig = plt.figure(frameon=False)
        image_name = 'result_{}.jpg'.format(int(time.time()))
        if os.path.exists('result.jpg'):
            fig.savefig(image_name, bbox_inches='tight', pad_inches=-0.1)

        else:
            fig.savefig('result.jpg', bbox_inches='tight', pad_inches=-0.1)

        ul.imshow(output, args.content_size, final=True, out='img_{}.jpg'.format(int(time.time())))

        # sphinx_gallery_thumbnail_number = 4
        plt.ioff()
        plt.show()

        # input_img2 = output.clone()
        # mix_output = style.run_style_transfer(cnn, content_img, input_img2, input_img,
        #                                 num_steps=args.iters, style_weight=args.style_weight,
        #                                 content_weight=args.content_weight)
        # fig = plt.figure(frameon=False)
        # ul.imshow(mix_output,args.content_size, final=True, out="img.jpg")
        # # sphinx_gallery_thumbnail_number = 4
        # plt.ioff()
        # plt.show()
        # fig.savefig("img2.jpg", bbox_inches='tight', pad_inches=-0.1)

    else:
        raise ValueError('Unknow experiment type')


if __name__ == "__main__":
    main()
