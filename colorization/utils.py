import torch
import functools
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
from models.pix2pix import G_net, D_net


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def network_initialization(args):
    netG = G_net(args)
    netD = D_net(args)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Using multi GPUs if you have
    if torch.cuda.device_count() != 0:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    # change device to set device (CPU or GPU)
    netG.to(args.device)
    netD.to(args.device)

    return netG, netD


def compute_gradient_penalty(args, D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random([real_samples.size(0), 1, 1, 1])).to(args.device)
    fake_samples.size()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(args.device)

    d_interpolates = D(interpolates)

    fake = torch.ones(d_interpolates.size()).to(args.device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = args.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def to_gray(args, imgs):
