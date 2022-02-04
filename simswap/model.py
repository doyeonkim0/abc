import torch
from faceswap_abc import FaceSwapInterface
from simswap.simswap import Generator_Adain_Upsample, Discriminator


class SimSwap(FaceSwapInterface):
    def __init__(self, gpu, args) -> None:
        super().__init__()
        self.set_models(gpu)
        self.set_optimizers(args)

    def set_models(self, gpu):
        self.G = Generator_Adain_Upsample(input_nc=3, output_nc=3, style_dim=512, n_blocks=9).cuda(gpu).train()
        self.D1 = Discriminator(input_nc=3).cuda(gpu).train()
        self.D2 = Discriminator(input_nc=3).cuda(gpu).train()

    def set_optimizers(self, args):
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=args.lr_G, betas=(0, 0.999))
        self.opt_D = torch.optim.Adam([*self.D1.parameters(), *self.D2.parameters()], lr=args.lr_D, betas=(0, 0.999))

    def load_next_batch(self):
        return super().load_next_batch()