from lib.loss import Loss, LossInterface


class SimSwapLoss(LossInterface):
    def get_loss_G(self, I_source, I_target, same_person, I_swapped, g_fake, g_real, id_swapped, id_source):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = Loss.get_hinge_loss(g_fake, True)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        
        # Identity loss
        if self.args.W_id:
            L_id = Loss.get_id_loss(id_source, id_swapped)
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # Attribute loss
        if self.args.W_attr:
            L_attr = Loss.get_attr_loss(I_target, I_swapped, self.args.batch_per_gpu)
            L_G += self.args.W_attr * L_attr
            self.loss_dict["L_attr"] = round(L_attr.item(), 4)

        # Reconstruction loss
        if self.args.W_recon:
            L_recon = Loss.get_L1_loss_with_same_person(I_swapped, I_target, same_person, self.args.batch_per_gpu)
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        
        # LPIPS loss
        if self.args.W_lpips:
            L_lpips = Loss.get_lpips_loss(I_swapped, I_target)
            L_G += self.args.W_lpips * L_lpips
            self.loss_dict["L_lpips"] = round(L_lpips.item(), 4)
        
        # Feature matching loss 
        if self.args.W_fm:
            L_fm = 0
            n_layers_D = 4
            num_D = 2
            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0 / num_D
            for i in range(0, n_layers_D):
                L_fm += D_weights * feat_weights * Loss.get_L1_loss(g_fake[i], g_real[i].detach())
            L_G += self.args.W_fm * L_fm
            self.loss_dict["L_fm"] = round(L_recon.item(), 4)

        self.loss_dict["L_G"] = round(L_G.item(), 4)

        return L_G

    def get_loss_D(self, d_real, d_fake):
        # Real 
        L_D_real = Loss.get_hinge_loss(d_real, True)
        L_D_fake = Loss.get_hinge_loss(d_fake, False)

        L_D = 0.5*(L_D_real.mean() + L_D_fake.mean())
        
        self.loss_dict["L_D_real"] = round(L_D_real.mean().item(), 4)
        self.loss_dict["L_D_fake"] = round(L_D_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
        