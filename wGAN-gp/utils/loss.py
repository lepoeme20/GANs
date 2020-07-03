import numpy as np
import torch
from torch import autograd


def compute_gradient_penalty(args, D, real_samples, fake_samples, cuda):
    """
    Computes gradient penalty
    """
    # Random weight term for interpolation between real and fake samples
    # Get random interpolation between real and fake samples
    alpha = torch.rand([real_samples.size(0), 1, 1, 1])
    if cuda:
        alpha = alpha.cuda(real_samples.device.index)

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    if cuda:
        interpolates = interpolates.cuda(real_samples.device.index)
    d_interpolates = D(interpolates)

    fake = torch.ones(d_interpolates.size())
    if cuda:
        fake = fake.cuda(real_samples.device.index)
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