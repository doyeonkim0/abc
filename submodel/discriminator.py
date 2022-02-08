import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator, self).__init__()

        kw = 4
        padw = 1
        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kw, stride=2, padding=padw),
            norm_layer(128), nn.LeakyReLU(0.2, True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=kw, stride=2, padding=padw),
            norm_layer(256), nn.LeakyReLU(0.2, True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=kw, stride=2, padding=padw),
            norm_layer(512), nn.LeakyReLU(0.2, True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kw, stride=1, padding=padw),
            norm_layer(512),
            nn.LeakyReLU(0.2, True)
        )

        if use_sigmoid:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)
            )

    def forward(self, input):
        out = []
        x = self.down1(input)
        out.append(x)
        x = self.down2(x)
        out.append(x)
        x = self.down3(x)
        out.append(x)
        x = self.down4(x)
        out.append(x)
        x = self.conv1(x)
        out.append(x)
        x = self.conv2(x)
        out.append(x)
        
        return out


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result
