import argparse
import sys

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="parser for PyTorch-Style-Transfer")
        subparsers = self.parser.add_subparsers(title="subcommands", dest="subcommand")
        optim_arg = subparsers.add_parser("optim",
                                    help="parser for optimization arguments")
        optim_arg.add_argument("--iters", type=int, nargs='?', const=1, default=400,
                                help="number of training iterations, default is 400")
        optim_arg.add_argument("--content-image", type=str, nargs='?',  const=1, default="images/dancing.jpg",
                                help="path to content image you want to stylize")
        optim_arg.add_argument("--style-image", type=str, nargs='?', const=1, default="images/rain_princess.jpg",
                                help="path to style-image")
        optim_arg.add_argument("--content-size", type=int, nargs='?', const=1, default=128,
                                help="factor for scaling down the content image")
        optim_arg.add_argument("--style-size", type=int, nargs='?', const=1, default=128,
                                help="size of style-image, default is the original size of style image")
        optim_arg.add_argument("--cuda", type=int, default=1,
                                help="set it to 1 for running on GPU, 0 for CPU")
        optim_arg.add_argument("--content-weight", type=float, nargs='?', const=1, default=1.0,
                                help="weight for content-loss, default is 1.0")
        optim_arg.add_argument("--style-weight", type=float, nargs='?', const=1,  default=5000.0,
                                help="weight for style-loss, default is 5000.0")
        optim_arg.add_argument("--second-style", type=str,
                                help="path to style-image")

    def parse(self):
        return self.parser.parse_args()
