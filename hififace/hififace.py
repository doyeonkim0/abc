import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from submodel import arcface
from deep3dmm import ParametricFaceModel, ReconNet
from lib.utils import AdaIN, weight_init


class HifiFace(nn.Module):
    def __init__(self):
        super(HifiFace, self).__init__()
        
        self.SAIE = ShapeAwareIdentityExtractor()
        self.SFFM = SemanticFacialFusionModule()
        self.E = Encoder()
        self.D = Decoder()

    def forward(self, I_s, I_t):
        
        # 3D Shape-Aware Identity Extractor
        v_sid, coeff_dict_fuse = self.SAIE(I_s, I_t)
        
        # Encoder
        z_latent, z_enc = self.E(I_t)

        # Decoder
        z_dec = self.D(z_latent, v_sid)

        # Semantic Facial Fusion Module
        I_r, I_low, M_low, M_high, z_dec, z_enc_, z_fuse = self.SFFM(z_enc, z_dec, v_sid, I_t)
        
        return I_r, I_low, M_low, M_high, z_dec, z_enc_, z_fuse, coeff_dict_fuse


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.InitConv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.AdaINResBlock1 = AdaINResBlock(512, 512, Resample_factor=1)
        self.AdaINResBlock2 = AdaINResBlock(512, 512, Resample_factor=1)
        self.AdaINResBlock3 = AdaINResBlock(512, 512, Resample_factor=2)
        self.AdaINResBlock4 = AdaINResBlock(512, 512, Resample_factor=2)
        self.AdaINResBlock5 = AdaINResBlock(512, 256, Resample_factor=2)

        self.apply(weight_init)

    def forward(self, feat, v_sid):
        feat1 = self.AdaINResBlock1(feat, v_sid) # 32x128x128
        feat2 = self.AdaINResBlock2(feat1, v_sid) # 64x64x64
        feat3 = self.AdaINResBlock3(feat2, v_sid) # 128x32x32
        feat4 = self.AdaINResBlock4(feat3, v_sid) # 256x16xx16
        z_dec = self.AdaINResBlock5(feat4, v_sid) # 512x8x8

        return z_dec


class ShapeAwareIdentityExtractor(nn.Module):
    def __init__(self):
        super(ShapeAwareIdentityExtractor, self).__init__()

        # face recognition model: arcface
        self.F_id = arcface.Backbone(50, 0.6, 'ir_se').eval()
        self.F_id.load_state_dict(torch.load('ptnn/arcface.pth', map_location="cuda"), strict=False)
        for param in self.F_id.parameters():
            param.requires_grad = False

        # 3D face reconstruction model
        self.net_recon = ReconNet()
        state_dict = torch.load("ptnn/deep3d.pth", map_location="cuda")
        self.net_recon.load_state_dict(state_dict['net_recon'])
        for param in self.net_recon.parameters():
            param.requires_grad = False
        self.facemodel = ParametricFaceModel(is_train=False)

        self.arcface_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    def forward(self, I_s, I_t):
        # id of Is
        v_id = self.get_id(I_s)

        # 3d params of Is
        coeff_dict_s = self.get_coeff3d(I_s)

        # 3d params of It
        coeff_dict_t = self.get_coeff3d(I_t)

        # fused 3d parms
        coeff_dict_fuse = coeff_dict_t.copy()
        coeff_dict_fuse["id"] = coeff_dict_s["id"]

        # concat all to obtain the 3D shape-aware identity(v_sid)
        v_sid = torch.cat([v_id, coeff_dict_fuse["id"], coeff_dict_fuse["exp"], coeff_dict_fuse["angle"]], dim=1)
        
        return v_sid, coeff_dict_fuse

    def get_id(self, I):
        v_id = self.F_id(F.interpolate(self.arcface_norm(I)[:, :, 16:240, 16:240], [112, 112], mode='bilinear', align_corners=True))
        return v_id

    def get_coeff3d(self, I):
        coeffs = self.net_recon(I[:, :, 16:240, 16:240])
        coeff_dict = self.facemodel.split_coeff(coeffs)

        return coeff_dict

    def get_lm3d(self, coeff_dict):
        
        # get 68 3d landmarks
        face_shape = self.facemodel.compute_shape(coeff_dict['id'], coeff_dict['exp'])
        rotation = self.facemodel.compute_rotation(coeff_dict['angle'])

        face_shape_transformed = self.facemodel.transform(face_shape, rotation, coeff_dict['trans'])
        face_vertex = self.facemodel.to_camera(face_shape_transformed)
        
        face_proj = self.facemodel.to_image(face_vertex)
        lm3d = self.facemodel.get_landmarks(face_proj)

        return lm3d


class SemanticFacialFusionModule(nn.Module):
    def __init__(self):
        super(SemanticFacialFusionModule, self).__init__()

        self.MaskResBlock = ResBlock(256, 1, Resample_factor=1)
        self.ResBlock = ResBlock(256, 256, Resample_factor=1)
        self.AdaINResBlock = AdaINResBlock(256, 259, Resample_factor=1)
        
        self.F_up = F_up()

        self.face_pool = nn.AdaptiveAvgPool2d((64, 64)).eval()


    def forward(self, z_enc, z_dec, v_sid, I_t):

        # z_enc 256 64 64
        # z_dec 256 64 64
        
        # M_low 1 64 64
        M_low = torch.sigmoid(self.MaskResBlock(z_dec))
        
        # z_fuse 256 64 64
        z_enc_ = self.ResBlock(z_enc)
        z_fuse = z_dec * M_low.repeat(1, 256, 1, 1) + z_enc_ * (1-M_low.repeat(1, 256, 1, 1))

        # I_out_low 256 64 64
        I_out_low = self.AdaINResBlock(z_fuse, v_sid)

        # I_low 3 64 64
        I_low = I_out_low[:, :3, :, :] * M_low.repeat(1, 3, 1, 1) + self.face_pool(I_t) * (1-M_low.repeat(1, 3, 1, 1))

        # I_out_high 3 256 256
        I_out_high, M_high = self.F_up(I_out_low[:, 3:, :, :])

        # I_r 3 256 256
        I_r = I_out_high * M_high.repeat(1, 3, 1, 1) + I_t * (1-M_high.repeat(1, 3, 1, 1))
        
        return I_r, I_low, M_low, M_high, z_dec, z_enc_, z_fuse


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, Resample_factor):
        super(ResBlock, self).__init__()

        self.InstanceNorm1 = nn.InstanceNorm2d(in_c)
        self.InstanceNorm2 = nn.InstanceNorm2d(out_c)
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.conv3x3_1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3x3_2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0, bias=False)

        if Resample_factor == 0.5:
            self.Resample = nn.AvgPool2d(3, stride=2, padding=1)

        else:
            self.Resample = nn.Upsample(scale_factor=Resample_factor, mode="bilinear", align_corners=False)

    def forward(self, feat):
        feat1 = self.InstanceNorm1(feat)
        feat1 = self.LReLU(feat1)
        feat1 = self.conv3x3_1(feat1)
        feat1 = self.Resample(feat1)
        feat1 = self.InstanceNorm2(feat1)
        feat1 = self.LReLU(feat1)
        feat1 = self.conv3x3_2(feat1)

        feat2 = self.conv1x1(feat)
        feat2 = self.Resample(feat2)

        return feat1 + feat2


class AdaINResBlock(nn.Module):
    def __init__(self, in_c, out_c, Resample_factor):
        super(AdaINResBlock, self).__init__()

        self.LReLU = nn.LeakyReLU(0.2, inplace=True)

        self.AdaIN1 = AdaIN(659, in_c)
        self.AdaIN2 = AdaIN(659, out_c)
        self.conv3x3_1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3x3_2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0, bias=False)

        if Resample_factor == 0.5:
            self.Resample = nn.AvgPool2d(3, stride=2, padding=1)

        else:
            self.Resample = nn.Upsample(scale_factor=Resample_factor, mode="bilinear", align_corners=False)
        self.count = 0

    def forward(self, feat, v_sid):
        feat1 = self.AdaIN1(feat, v_sid)
        feat1 = self.LReLU(feat1)
        feat1 = self.conv3x3_1(feat1)
        feat1 = self.Resample(feat1)
        feat1 = self.AdaIN2(feat1, v_sid)
        feat1 = self.LReLU(feat1)
        feat1 = self.conv3x3_2(feat1)

        feat2 = self.conv1x1(feat)
        feat2 = self.Resample(feat2)

        return feat1 + feat2


class F_up(nn.Module):
    def __init__(self):
        super(F_up, self).__init__()
        self.ResBlock_image1 = ResBlock(256, 256, Resample_factor=2)
        self.ResBlock_image2 = ResBlock(256, 256, Resample_factor=2)
        self.ResBlock_image3 = ResBlock(256, 4, Resample_factor=1)
        
        self.sigmoid = nn.Sigmoid()
        # self.Tanh = nn.Tanh()

    def forward(self, I_out_low):
        feat_image1 = self.ResBlock_image1(I_out_low)
        feat_image2 = self.ResBlock_image2(feat_image1)
        feat_image3 = self.ResBlock_image3(feat_image2)
        image = feat_image3[:, :3, :, :]
        mask = self.sigmoid(feat_image3[:, 3, :, :].unsqueeze(1))

        return image, mask


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.InitConv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.ResBlock1 = ResBlock(64, 128, Resample_factor=0.5)
        self.ResBlock2 = ResBlock(128, 256, Resample_factor=0.5)
        self.ResBlock3 = ResBlock(256, 512, Resample_factor=0.5)
        self.ResBlock4 = ResBlock(512, 512, Resample_factor=0.5)
        self.ResBlock5 = ResBlock(512, 512, Resample_factor=0.5)
        self.ResBlock6 = ResBlock(512, 512, Resample_factor=1)
        self.ResBlock7 = ResBlock(512, 512, Resample_factor=1)

        self.apply(weight_init)

    def forward(self, It):
        feat0 = self.InitConv(It) # 32x128x128
        feat1 = self.ResBlock1(feat0) # 32x128x128
        feat2 = self.ResBlock2(feat1) # 64x64x64
        feat3 = self.ResBlock3(feat2) # 128x32x32
        feat4 = self.ResBlock4(feat3) # 256x16xx16
        feat5 = self.ResBlock5(feat4) # 512x8x8
        feat6 = self.ResBlock6(feat5) # 1024x4x4
        feat7 = self.ResBlock7(feat6) # 1024x4x4

        return feat7, feat2
