import torch
from lib import checkpoint, utils
from lib.faceswap import FaceSwapInterface
from loss import FaceShifterLoss
from AEI_Net import AEI_Net
from submodel.discriminator import MultiscaleDiscriminator


class FaceShifter(FaceSwapInterface):
    def initialize_models(self):
        self.G = AEI_Net(c_id=512).cuda(self.gpu).train()
        self.D = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).cuda(self.gpu).train()

    def set_multi_GPU(self):
        utils.setup_ddp(self.gpu, self.args.gpu_num)

        # Data parallelism is required to use multi-GPU
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
        self._loss_collector = FaceShifterLoss(self.args)

    def train_step(self):
        I_source, I_target, same_person = self.load_next_batch()
        
        ###########
        # train G #
        ###########

        Y, I_t_attr, I_s_id = self.G(I_source, I_target)
        Y_attr = self.G.get_attr(Y)
        Y_id = self.G.get_id(Y)
        d_adv = self.D(Y)
        d_true = self.D(I_source)
        d_fake = self.D(Y.detach())

        loss_G = self.loss_collector.get_loss_G(I_target, Y, I_t_attr, I_s_id, Y_attr, Y_id, d_adv, same_person)
        utils.update_net(self.opt_G, loss_G)

        ###########
        # train D #
        ###########

        loss_D = self.loss_collector.get_loss_D(d_true, d_fake)
        utils.update_net(self.opt_D, loss_D)

        return [I_source, I_target, Y]

    def save_checkpoint(self, step):
        checkpoint.save_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.save_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)

    @property
    def loss_collector(self):
        return self._loss_collector
        