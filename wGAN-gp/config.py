import multiprocessing
import argparse
import pytorch_lightning as pl

def set_parser(parser):
    # Networks hyper-parameters
    net_args = parser.add_argument_group('Networks')
    net_args.add_argument('--ngf', type=int, default=64, help='G # channels')
    net_args.add_argument('--ndf', type=int, default=64, help='D # channels')
    net_args.add_argument('--latent-dim', type=int, default=100)

    # Optimization hyper-parameters
    opt_args = parser.add_argument_group('Optimization')
    opt_args.add_argument("--epochs", type=int, default=1000, help="number of epochs for training")
    opt_args.add_argument("--batch_size", type=int, default=64, help="size of the mini-batch")
    opt_args.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    opt_args.add_argument(
        "--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    opt_args.add_argument(
        "--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    opt_args.add_argument(
        "--lambda_gp", type=int, default=10, help="Loss weight for gradient penalty")

    #
    base_args = parser.add_argument_group('others')
    base_args.add_argument(
        "--dataset", type=str, default='cifar10', help='[Mnist, cifar10, cifar100]'
        )
    base_args.add_argument(
        "--n_cpu", type=int, default=multiprocessing.cpu_count()-1,
        help="number of cpu threads to use during batch generation"
        )
    base_args.add_argument("--image-size", type=int, default=32, help="Image size")
    base_args.add_argument("--channels", type=int, default=3, help="Image channels")
    base_args.add_argument(
        "--data-root-path", type=str, default='/media/lepoeme20/Data/basics', help='data path'
    )
    return parser

def get_config():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = set_parser(parser)
    args, _ = parser.parse_known_args()

    return args