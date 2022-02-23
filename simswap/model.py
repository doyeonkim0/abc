import torch
from lib import checkpoint, utils
from lib.faceswap import FaceSwapInterface
from simswap.simswap import Generator_Adain_Upsample
from submodel.discriminator import Discriminator
from simswap.loss import SimSwapLoss


class SimSwap(FaceSwapInterface):
    def __init__(self, args, gpu):
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        super().__init__(args, gpu)

    def initialize_models(self):
        self.G = Generator_Adain_Upsample(input_nc=3, output_nc=3, style_dim=512, n_blocks=9).cuda(self.gpu).train()
        self.D = Discriminator(input_nc=3).cuda(self.gpu).train()

    def set_multi_GPU(self):
        utils.setup_ddp(self.gpu, self.args.gpu_num)

        # Data parallelism is required to use multi-GPU
        self.G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True).module
        self.D = torch.nn.parallel.DistributedDataParallel(self.D, device_ids=[self.gpu]).module
        
    def load_checkpoint(self, step=-1):
        checkpoint.load_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.load_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)

    def set_optimizers(self):
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.args.lr_G, betas=(0, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.args.lr_D, betas=(0, 0.999))

    def set_loss_collector(self):
        self._loss_collector = SimSwapLoss(self.args)

    def train_step(self):
        I_source, I_target, same_person = self.load_next_batch()

        ###########
        # Train G #
        ###########

        G_list = [I_source, I_target, same_person]

        # Run G to swap identity from source to target image
        I_swapped = self.G(I_source, I_target)
        G_list += [I_swapped]

        # Run D
        g_fake = self.D(I_swapped)
        g_real = self.D(I_target)
        G_list += [g_fake, g_real]
        
        # Run ArcFace to extract identity vectors from swapped and source image
        id_swapped = self.G.get_id(I_swapped)
        id_source = self.G.get_id(I_source)
        G_list += [id_swapped, id_source]

        loss_G = self.loss_collector.get_loss_G(*G_list)
        utils.update_net(self.opt_G, loss_G)

        ###########
        # Train D #
        ###########

        D_list = []

        # D_Real
        d_real = self.D(I_target)
        D_list += [d_real]

        # D_Fake
        d_fake = self.D(I_swapped.detach())
        D_list += [d_fake]

        loss_D = self.loss_collector.get_loss_D(*D_list)
        utils.update_net(self.opt_D, loss_D)

        return [I_source, I_target, I_swapped]

    def save_checkpoint(self, step):
        checkpoint.save_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.save_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)

    def save_image(self, result, step):
        utils.save_image(self.args, step, "imgs", result)
        
    @property
    def loss_collector(self):
        return self._loss_collector
