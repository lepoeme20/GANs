import argparse
import os
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.autograd as autograd
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
parser.add_argument("--lambda_gp", type=int, default=10, help="Loss weight for gradient penalty")
parser.add_argument("--data_path", type=str, default='../data/cifar10', help="root path for dataset")
opt, _ = parser.parse_known_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

device = ('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, kernal_size, stride, padding):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=kernal_size,
                                         stride=stride, padding=padding),
                      nn.BatchNorm2d(out_feat, 0.8),
                      nn.ReLU(inplace=True)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 8 * opt.latent_dim, 4, 1, 0),
            *block(8 * opt.latent_dim, 4 * opt.latent_dim, 4, 2, 1),
            *block(4 * opt.latent_dim, 2 * opt.latent_dim, 4, 2, 1),
            nn.ConvTranspose2d(2 * opt.latent_dim, opt.channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        # img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(opt.channels, 2 * opt.latent_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * opt.latent_dim, 4 * opt.latent_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * opt.latent_dim, 8 * opt.latent_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear = nn.Linear(4 * 4 * 4 * opt.latent_dim, 1)
        self.output = nn.Conv2d(8 * opt.latent_dim, 1, 4, 1, 0)

    def forward(self, img):
        out = self.model(img)
        # out = out.view(-1, 4 * 4 * 4 * opt.latent_dim)
        out = self.output(out)
        return out


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Using DataParallel if you have more than one GPU
if torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

# Set models device
generator.to(device)
discriminator.to(device)

# Configure data loader
os.makedirs(opt.data_path, exist_ok=True)
train_dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(
        root=opt.data_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def compute_gradient_penalty(D, real_samples, fake_samples):
    """
    Computes gradient penalty
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random([real_samples.size(0), 1, 1, 1])).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)

    d_interpolates = D(interpolates)

    fake = torch.ones(d_interpolates.size()).to(device)
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
    gradient_penalty = opt.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

batches_done = 0

# Set for backward propagation
one = torch.tensor(1, dtype=torch.float).to(device)
mone = one * -1

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(train_dataloader):

        # Configure input
        real_imgs = imgs.to(device)

        for p in discriminator.parameters():
            p.requires_grad = True

        # ---------------------
        #  (1) Update D
        # ---------------------

        optimizer_D.zero_grad()

        # Real images
        real_validity = discriminator(real_imgs)
        d_loss_real = real_validity.mean()
        d_loss_real.backward(mone)

        # Sample noise as generator input
        z = torch.rand(imgs.size(0), opt.latent_dim, 1, 1).to(device)
        # Generate a batch of images
        fake_imgs = generator(z)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        d_loss_fake = fake_validity.mean()
        d_loss_fake.backward(one)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        gradient_penalty.backward()

        # Adversarial loss
        d_loss = d_loss_fake - d_loss_real + gradient_penalty

        optimizer_D.step()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  (2) Update G
            # -----------------
            for p in discriminator.parameters():
                p.requires_grad = False

            optimizer_G.zero_grad()
            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = fake_validity.mean()
            g_loss.backward(mone)
            g_cost = -g_loss

            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), g_cost.item())
            )

            if batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d%s.png" % (batches_done, "_fake"), nrow=5, normalize=True)
                # save_image(real_imgs.data[:25], "images/%d%s.png" % (batches_done, "_real"), nrow=5, normalize=True)

            batches_done += opt.n_critic

    # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
    #     # Save model checkpoints
    #     torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
    #     torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
