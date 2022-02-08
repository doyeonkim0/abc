import submodel.arcface as arcface
import torch
from torch import nn
import torch.nn.functional as F


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, padding_type, activation=nn.ReLU(True), style_dim=512):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding = p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = AdaIN(style_dim, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = AdaIN(style_dim, dim)


    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out


class Generator_Adain_Upsample(nn.Module):
    def __init__(self, input_nc, output_nc, style_dim=512, n_blocks=6, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Generator_Adain_Upsample, self).__init__()
        activation = nn.ReLU(True)

        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
                                         norm_layer(64), activation)
        ### downsample
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   norm_layer(128), activation)
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   norm_layer(256), activation)
        self.down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   norm_layer(512), activation)
        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(512, padding_type=padding_type, activation=activation, style_dim=style_dim)]
        self.BottleNeck = nn.Sequential(*BN)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), activation
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), activation
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), activation
        )
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
                                        nn.Tanh())

        # face recognition model: arcface
        self.arcface = arcface.Backbone(50, 0.6, 'ir_se').eval()
        self.arcface.load_state_dict(torch.load('ptnn/model_ir_se50.pth', map_location="cuda"), strict=False)
        for param in self.arcface.parameters():
            param.requires_grad = False

    def get_id(self, I):
        return self.arcface(F.interpolate(I[:, :, 16:240, 16:240], [112, 112], mode='bilinear', align_corners=True))

    def forward(self, I_source, I_target):

        id_source = self.get_id(I_source)

        x = I_target  # 3*224*224

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        x = self.down3(skip3)

        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, id_source)

        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.last_layer(x)
        # x = (x + 1) / 2
        return x
        