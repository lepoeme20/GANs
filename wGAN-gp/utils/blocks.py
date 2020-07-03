from torch import nn

class ResBlock(nn.Module):
    def __init__(self, args, dim):
        super(ResBlock, self).__init__()
        self.channels = int(dim / 2)
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [
            nn.Conv2d(dim, self.channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(True)]

        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.channels, self.channels, kernel_size=3),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(True)]

        conv_block += [
            nn.Conv2d(self.channels, dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class scaling(nn.Module):
    def __init__(self, in_ch, out_ch, up=True):
        super(scaling, self).__init__()
        if up:
            self.scaling = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(
                    in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                # nn.ConvTranspose2d(
                #     in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            )

        else:
            self.scaling = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
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