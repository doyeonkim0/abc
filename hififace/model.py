
import torch
from lib import checkpoint, utils
from lib.faceswap import FaceSwapInterface
from loss import HifiFaceLoss
from hififace import HifiFace
from submodel.discriminator import StarGANv2Discriminator
from submodel.segmentation import Segmentation


class HifiFace(FaceSwapInterface):
    def __init__(self, args, gpu):
        self.upsample = torch.nn.Upsample(scale_factor=4).to(gpu).eval()
        super().__init__(args, gpu)

    def initialize_models(self):
        self.G = HifiFace().cuda(self.gpu).train()
        self.D = StarGANv2Discriminator().cuda(self.gpu).train()

    def set_multi_GPU(self):
        utils.setup_ddp(self.gpu, self.args.gpu_num)

        G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True)
        D = torch.nn.parallel.DistributedDataParallel(self.D, device_ids=[self.gpu])
        self.G = G.module
        self.D = D.module

    def load_checkpoint(self, step=-1):
        checkpoint.load_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.load_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)

    def set_optimizers(self):
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.args.lr_G, betas=(0, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.args.lr_D, betas=(0, 0.999))

    def set_loss_collector(self):
        self.loss_collector = HifiFaceLoss(self.args)

    def train_step(self):
        I_s, I_t, same_person = self.load_next_batch()
        I_r, I_low, M_low, M_high, z_dec, z_enc, z_fuse, c_fuse = self.G(I_s, I_t)
        I_low_256 = self.upsample(I_low)
        
        # Arcface 
        v_id_s = self.G.SAIE.get_id(I_s)
        v_id_r = self.G.SAIE.get_id(I_r)
        v_id_low = self.G.SAIE.get_id(I_low_256)

        # 3D landmarks
        q_r = self.G.SAIE.get_lm3d(self.G.SAIE.get_coeff3d(I_r))
        q_low = self.G.SAIE.get_lm3d(self.G.SAIE.get_coeff3d(I_low_256))
        q_fuse = self.G.SAIE.get_lm3d(c_fuse)

        # Segmentation mask
        M_tar = Segmentation.dilate(Segmentation.get_mask(I_t))

        # Cycle image
        I_cycle = self.G(I_t, I_r)[0]

        # Discriminator
        d_adv = self.D(I_r)
        I_t.requires_grad_()
        d_true = self.D(I_t)
        d_fake = self.D(I_r.detach())

        ###########
        # train G #
        ###########

        loss_G = self.loss_collector.get_loss_G(
            I_t, I_r, I_low, v_id_s, v_id_r, v_id_low, M_tar, M_high, M_low, 
            q_r, q_low, q_fuse, I_cycle, d_adv, same_person)
        utils.update_net(self.opt_G, loss_G)

        ###########
        # train D #
        ###########

        loss_D = self.loss_collector.get_loss_D(I_t, d_true, d_fake)
        utils.update_net(self.opt_D, loss_D)

        return [I_s]

    def save_checkpoint(self, step):
        checkpoint.save_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.save_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)   
