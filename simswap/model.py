import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib import checkpoint, utils
from lib.faceswap import FaceSwapInterface
from lib.dataset import FaceDataset
from simswap.simswap import Generator_Adain_Upsample, Discriminator
from loss import SimSwapLoss


class SimSwap(FaceSwapInterface):
    def __init__(self, args, gpu):
        self.args = args
        self.gpu = gpu
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.model_name = 'G'

    def initialize_models(self):
        self.G = Generator_Adain_Upsample(input_nc=3, output_nc=3, style_dim=512, n_blocks=9).cuda(self.gpu).train()
        self.D1 = Discriminator(input_nc=3).cuda(self.gpu).train()
        self.D2 = Discriminator(input_nc=3).cuda(self.gpu).train()

    def set_dataset(self):
        self.dataset = FaceDataset(self.args.dataset_list, same_prob=self.args.same_prob)

    def set_data_iterator(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.dataset) if self.args.use_mGPU else None
        dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size, pin_memory=True, sampler=sampler, num_workers=8, drop_last=True)
        self.iterator = iter(dataloader)

    def set_multi_GPU(self):
        utils.setup_ddp(self.gpu, self.args.gpu_num)

        # Data parallelism is required to use multi-GPU
        G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True)
        D1 = torch.nn.parallel.DistributedDataParallel(self.D1, device_ids=[self.gpu])
        D2 = torch.nn.parallel.DistributedDataParallel(self.D2, device_ids=[self.gpu])
        self.G = G.module
        self.D1 = D1.module
        self.D2 = D2.module
        
    def load_checkpoint(self, step=-1):
        checkpoint.load_checkpoint(self.args, self.G, name=self.model_name, global_step=step)

    def set_optimizers(self):
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.args.lr_G, betas=(0, 0.999))
        self.opt_D = torch.optim.Adam([*self.D1.parameters(), *self.D2.parameters()], lr=self.args.lr_D, betas=(0, 0.999))

    def set_loss_collector(self):
        self.loss_collector = SimSwapLoss(self.args)

    def load_next_batch(self):
        I_source, I_target, same_person = next(self.iterator)
        I_source, I_target, same_person = I_source.to(self.gpu), I_target.to(self.gpu), same_person.to(self.gpu)
        return I_source, I_target, same_person

    def train_step(self):
        I_source, I_target, same_person = self.load_next_batch()

        ###########
        # Train G #
        ###########

        G_list = [I_source, I_target, same_person]

        # Run G to swap identity from source to target image
        I_swapped = self.G(I_source, I_target)
        G_list += [I_swapped]

        # Downsample
        I_swapped_downsampled = self.downsample(I_swapped)
        I_target_downsampled = self.downsample(I_target)

        # Run D
        g_fake1 = self.D1.forward(I_swapped)
        g_fake2 = self.D2.forward(I_swapped_downsampled)
        g_real1 = self.D1.forward(I_target)
        g_real2 = self.D2.forward(I_target_downsampled)
        G_list += [g_fake1, g_fake2, g_real1, g_real2]
        
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
        d_real1 = self.D1.forward(I_target)
        d_real2 = self.D2.forward(I_target_downsampled)
        D_list += [d_real1, d_real2]

        # D_Fake
        d_fake1 = self.D1.forward(I_swapped.detach())
        d_fake2 = self.D2.forward(I_swapped_downsampled.detach())
        D_list += [d_fake1, d_fake2]

        loss_D = self.loss_collector.get_loss_D(*D_list)
        utils.update_net(self.opt_D, loss_D)

        return [I_source, I_target, I_swapped]

    def save_image(self, result, step):
        utils.save_image(self.args, step, "imgs", result)

    def save_checkpoint(self, step):
        checkpoint.save_checkpoint(self.args, self.G, step, name=self.model_name)
