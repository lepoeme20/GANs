import torch
import argparse

def parser_setting(parser):
    default_parser = parser.add_argument_group('Default')
    default_parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    default_parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    default_parser.add_argument("--lambda", type=int, default=100, help="Weight of pixel-wise loss")
    default_parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
    default_parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    default_parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    default_parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    default_parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    default_parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    default_parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    default_parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    default_parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    default_parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    default_parser.add_argument(
        "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
    )
    default_parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")


def get_config():
    parser = argparse.ArgumentParser()
    default_parser = parser_setting(parser)
    args, _ = default_parser.parse_known_args()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return args
