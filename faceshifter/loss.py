import time
from lib.loss import Loss, LossInterface


class FaceShifterLoss(LossInterface):
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self._loss_dict = {}

    def get_loss_G(self, I_t, Y, I_t_attr, I_s_id, Y_attr, Y_id, d_adv, same_person):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = Loss.get_hinge_loss(d_adv, True)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        
        # Id loss
        if self.args.W_id:
            L_id = Loss.get_id_loss(I_s_id.detach(), Y_id)
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # Attribute loss
        if self.args.W_attr:
            L_attr = Loss.get_attr_loss(I_t_attr, Y_attr, self.args.batch_per_gpu)
            L_G += self.args.W_attr * L_attr
            self.loss_dict["L_attr"] = round(L_attr.item(), 4)

        # Reconstruction loss
        if self.args.W_recon:
            L_recon = Loss.get_L2_loss_with_same_person(Y, I_t, same_person, self.args.batch_per_gpu)
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        
        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_loss_D(self, d_real, d_fake):
        L_real = Loss.get_hinge_loss(d_real, True)
        L_fake = Loss.get_hinge_loss(d_fake, False)
        L_D = 0.5*(L_real.mean() + L_fake.mean())
        
        self.loss_dict["L_real"] = round(L_real.mean().item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

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
        