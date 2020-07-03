import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import pytorch_lightning as pl
from collections import OrderedDict
from utils.blocks import conv, scaling, ResBlock
from utils.loss import compute_gradient_penalty
from utils.dataloader import get_dataloader


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.ngf = args.ngf
        # self.init_size = args.image_size // 4
        # self.stem = nn.Linear(args.latent_dim, 2 * args.ngf * (self.init_size ** 2))

        def stem(in_feat, out_feat, kernel_size, strid, padding):
            layers = [
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size, strid, padding, bias=False),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            return layers

        def block(in_feat, out_feat, kernel_size, stride, padding):
            layers = [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_feat, 0.8),
                nn.LeakyReLU(0.2, inplace=True)]
            return layers

        def last_block(in_feat, out_feat):
            layer = [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_feat, out_feat, 3, 1, 1, bias=False),
                nn.Tanh()
            ]
            return layer

        self.stem = nn.Sequential(*stem(args.latent_dim, 8 * args.ngf, 4, 1, 0))
        self.model = nn.Sequential(
            *block(8 * args.ngf, 4 * args.ngf, 3, 1, 1),
            *block(4 * args.ngf, 2 * args.ngf, 3, 1, 1),
            *last_block(2 * args.ngf, args.channels),
        )

    def forward(self, z):
        init_img = self.stem(z)
        # init_img = init_img.view(z.size(0), 2 * self.ngf, self.init_size, self.init_size)
        img = self.model(init_img)

        return img

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters):
            block = [
                nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                ]
            return block

        self.model = nn.Sequential(
            *discriminator_block(args.channels, args.ndf),
            *discriminator_block(args.ndf, args.ndf * 2),
            *discriminator_block(args.ndf * 2, args.ndf * 4),
            *discriminator_block(args.ndf * 4, args.ndf * 8), # 2 x 2
        )

        # The height and width of downsampled image
        ds_size = args.image_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(args.ndf * 8 * ds_size ** 2, 1),
            nn.Sigmoid()
            )

    def forward(self, img):
        out = self.model(img)
        out = out.view(img.size(0), -1)
        validity = self.adv_layer(out)
        return validity


class wGANGP(pl.LightningModule):
    def __init__(self, hparams):
        super(wGANGP, self).__init__()
        self.hparams = hparams

        # networks
        self.generator, self.discriminator = self.__network_initialization(hparams)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, z):
        return self.generator(z)

    def criterion_gan(self, y_hat, y):
        # y_hat = y_hat.squeeze(2).squeeze(2)
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_idx):
        imgs, labels = batch
        self.last_imgs = imgs

        # train generator
        if optimizer_idx == 0:
            z = torch.rand(imgs.size(0), self.hparams.latent_dim, 1, 1)
            valid = torch.ones((imgs.size(0), 1))

            if self.on_gpu:
                z = z.cuda(imgs.device.index)
                valid = valid.cuda(imgs.device.index)

            fake_imgs = self(z)

            # g loss
            fake_validity = self.discriminator(fake_imgs)
            g_loss = self.criterion_gan(fake_validity, valid)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            valid = torch.ones((imgs.size(0), 1))
            fake = torch.zeros((imgs.size(0), 1))
            z = torch.rand(imgs.size(0), self.hparams.latent_dim, 1, 1)

            if self.on_gpu:
                z = z.cuda(imgs.device.index)
                valid = valid.cuda(imgs.device.index)
                fake = fake.cuda(imgs.device.index)

            # d loss
            real_validity = self.discriminator(imgs)
            real_loss = self.criterion_gan(real_validity, valid)

            fake_imgs = self(z)
            fake_validity = self.discriminator(fake_imgs.detach())
            fake_loss = self.criterion_gan(fake_validity, fake)

            gp_loss = compute_gradient_penalty(
                self.hparams, self.discriminator, imgs, fake_imgs.detach(), self.on_gpu)
            d_loss = (real_loss + fake_loss)/2 + gp_loss
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def on_epoch_end(self):
        z = torch.rand(self.last_imgs.size(0), self.hparams.latent_dim, 1, 1)
        if self.on_gpu:
            z = z.cuda(self.last_imgs.device.index)

        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs[:16], normalize=True, nrow=4)
        self.logger.experiment.add_image(f'images', grid, self.current_epoch)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        n_critic = 5

        return(
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': n_critic},
        )

    def train_dataloader(self):
        trn_dataloader, _, _ = get_dataloader(self.hparams)
        return trn_dataloader

    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def __network_initialization(self, args):
        netG = Generator(args)
        netD = Discriminator(args)

        # network initialization
        netG.apply(self.__weights_init)
        netD.apply(self.__weights_init)

        return netG, netD