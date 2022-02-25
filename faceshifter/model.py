import torch
from lib import checkpoint, utils
from lib.faceswap import FaceSwapInterface
from faceshifter.loss import FaceShifterLoss
from faceshifter.faceshifter import AEI_Net
from submodel.discriminator import MultiscaleDiscriminator


class FaceShifter(FaceSwapInterface):
    def initialize_models(self):
        self.G = AEI_Net(c_id=512).cuda(self.gpu).train()
        self.D = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).cuda(self.gpu).train()

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
        self._loss_collector = FaceShifterLoss(self.args)

    def train_step(self):
        I_source, I_target, same_person = self.load_next_batch()
        
        ###########
        # train G #
        ###########

        Y, I_target_attr, I_source_id = self.G(I_source, I_target)
        Y_attr = self.G.get_attr(Y)
        Y_id = self.G.get_id(Y)
        d_adv = self.D(Y)

        G_dict = {
            "I_source": I_source,
            "I_target": I_target, 
            "same_person": same_person, 
            "I_target_attr": I_target_attr,
            "I_source_id": I_source_id,
            "Y": Y,
            "Y_attr": Y_attr,
            "Y_id": Y_id,
            "d_adv": d_adv
        }

        loss_G = self.loss_collector.get_loss_G(G_dict)
        utils.update_net(self.opt_G, loss_G)

        ###########
        # train D #
        ###########

        d_real = self.D(I_source)
        d_fake = self.D(Y.detach())

        D_dict = {
            "d_real": d_real,
            "d_fake": d_fake,
        }
        
        loss_D = self.loss_collector.get_loss_D(D_dict)
        utils.update_net(self.opt_D, loss_D)

        return [I_source, I_target, Y]

    def save_image(self, result, step):
        utils.save_image(self.args, step, "imgs", result)
        
    def save_checkpoint(self, step):
        checkpoint.save_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.save_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)

    @property
    def loss_collector(self):
        return self._loss_collector
        