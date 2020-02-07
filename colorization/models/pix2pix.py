import torch.nn as nn


class G_net(nn.Module):
    def __init__(self, args):
        super(G_net, self).__init__()
        self.args = args

        def main_block():
            layers = [nn.ConvTranspose2d(1, 2 * args.ndf, kernel_size=4, stride=2, padding=1),
                           nn.BatchNorm2d(2 * args.ngf), nn.ReLU(True),
                           conv(2 * args.ngf, 4 * args.ngf, init=True),
                           scaling(4 * args.ngf, 8 * args.ngf, up=False),
                           scaling(8 * args.ngf, 8 * args.ngf, up=False)]

            # Start Generator
            model = []
            for i in range(6):
                model += [ResBlock(args, 8 * args.ngf)]
            layers.append(nn.Sequential(*model))
            layers.append(scaling(8 * args.ngf, 4 * args.ngf, up=True))
            layers.append(scaling(4 * args.ngf, 2 * args.ngf, up=True))
            layers.append(conv(2 * args.ngf, args.channels, init=False))
            layers.append(nn.Tanh())

            return nn.Sequential(*layers)

        self.main_block = main_block()

    def forward(self, img):
        output = self.main_block(img)

        return output


class D_net(nn.Module):
    def __init__(self, args):
        super(D_net, self).__init__()

        def block(in_channels, out_channels, kernel_size, stride, padding):
            layers = [nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding),
                      nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.model = nn.Sequential(
            *block(args.channels, 2 * args.ndf, kernel_size=4, stride=2, padding=1),
            *block(2 * args.ndf, 4 * args.ndf, kernel_size=4, stride=2, padding=1),
            *block(4 * args.ndf, 8 * args.ndf, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(8 * args.ndf, 1, 4, 1, 0)
        )

    def forward(self, img):
        out = self.model(img)
        return out


# Define a Resnet block
class ResBlock(nn.Module):
    def __init__(self, args, dim):
        super(ResBlock, self).__init__()
        self.channels = args.ngf
        self.conv_block = self.build_conv_block(dim)
        self.relu = nn.ReLU(True)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.Conv2d(dim, self.channels, kernel_size=1, padding=0),
                       nn.BatchNorm2d(self.channels),
                       nn.ReLU(True)]

        conv_block += [nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
                       nn.BatchNorm2d(self.channels),
                       nn.ReLU(True)]

        conv_block += [nn.Conv2d(self.channels, dim, kernel_size=1, padding=0),
                       nn.BatchNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        _out = x + self.conv_block(x)
        out = self.relu(_out)
        return out


class scaling(nn.Module):
    def __init__(self, in_ch, out_ch, up=True):
        super(scaling, self).__init__()
        if up:
            self.scaling = nn.Sequential(
                # nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.Conv2d(in_ch, out_ch,
                #           kernel_size=3, stride=1,
                #           padding=1, bias=use_bias),
                nn.ConvTranspose2d(in_ch, out_ch,
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            )

        else:
            self.scaling = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3,
                          stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            )

    def forward(self, x):
        out = self.scaling(x)
        return out


class conv(nn.Module):
    def __init__(self, in_ch, out_ch, init=True):
        super(conv, self).__init__()
        if init:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            )
        else:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
                nn.Tanh()
            )

    def forward(self, tensor):
        out = self.block(tensor)
        return out
