import time
import torch
from lib.loss import Loss, LossInterface


class HifiFaceLoss(LossInterface):
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self._loss_dict = {}
        self.face_pool = torch.nn.AdaptiveAvgPool2d((64, 64)).to("cuda").eval()

    def get_loss_G(self, I_t, I_r, I_low, v_id_s, v_id_r, v_id_low, M_tar, M_high, M_low, q_r, q_low, q_fuse, I_cycle, d_adv, same_person):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = Loss.get_BCE_loss(d_adv, True)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        
        # Shape loss
        if self.args.W_shape:
            L_shape = Loss.get_L1_loss(q_fuse, q_r) 
            L_shape += Loss.get_L1_loss(q_fuse, q_low)
            L_G += self.args.W_shape * L_shape/68
            self.loss_dict["L_shape"] = round(L_shape.item(), 4)

        # Id loss
        if self.args.W_id:
            L_id = Loss.get_id_loss(v_id_s.detach(), v_id_r)
            L_id += Loss.get_id_loss(v_id_s.detach(), v_id_low)
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # Reconstruction loss
        if self.args.W_recon:
            L_recon = Loss.get_L1_loss_with_same_person(I_r, I_t, same_person, self.args.batch_size)
            L_recon += Loss.get_L1_loss_with_same_person(I_low, self.face_pool(I_t), same_person, self.args.batch_size)
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        
        # Mask loss
        if self.args.W_seg:
            L_seg = Loss.get_L1_loss(self.face_pool(M_tar), M_low)
            L_seg += Loss.get_L1_loss(M_tar, M_high)
            L_G += self.args.W_seg * L_seg
            self.loss_dict["L_seg"] = round(L_seg.item(), 4)

        # Cycle loss
        if self.args.W_cycle:
            L_cycle = Loss.get_L1_loss(I_t, I_cycle)
            L_G += self.args.W_cycle * L_cycle
            self.loss_dict["L_cycle"] = round(L_cycle.item(), 4)

        # LPIPS loss
        if self.args.W_lpips:
            L_lpips = Loss.get_lpips_loss(I_r, I_t)
            L_lpips += Loss.get_lpips_loss(I_low, self.face_pool(I_t))
            L_G += self.args.W_lpips * L_lpips
            self.loss_dict["L_lpips"] = round(L_lpips.item(), 4)

        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_loss_D(self, I_t, d_true, d_fake):
        L_true = Loss.get_BCE_loss(d_true, True)
        L_fake = Loss.get_BCE_loss(d_fake, False)
        L_reg = Loss.get_r1_reg(d_true, I_t)
        L_D = L_true + L_fake + L_reg
        
        self.loss_dict["L_D"] = round(L_D.item(), 4)
        self.loss_dict["L_true"] = round(L_true.mean().item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)

        return L_D
    
    def print_loss(self, global_step):
        seconds = int(time.time() - self.start_time)
        print("")
        print(f"[ {self.format_time(seconds)} ]")
        print(f'steps: {global_step:06} / {self.args.max_step}')
        print(f'lossD: {self.loss_dict["L_D"]} | lossG: {self.loss_dict["L_G"]}')

    @property
    def loss_dict(self):
        return self._loss_dict
