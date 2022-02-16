import abc
from submodel.lpips import LPIPS
import torch
import torch.nn.functional as F


class LossInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_loss_G(self):
        pass

    @abc.abstractmethod
    def get_loss_D(self):
        pass

    @abc.abstractmethod
    def print_loss(self):
        pass

    @property
    @abc.abstractmethod
    def loss_dict(self):
        pass

    def format_time(self, seconds):
        return f"{seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s"


class Loss:
    L1 = torch.nn.L1Loss().to("cuda")
    L2 = torch.nn.MSELoss().to("cuda")

    def get_id_loss(a, b):
        return (1 - torch.cosine_similarity(a, b, dim=1)).mean()

    @classmethod
    def get_lpips_loss(cls, a, b):
        if not hasattr(cls, 'lpips'):
            cls.lpips = LPIPS().eval().to("cuda")
        return cls.lpips(a, b)

    @classmethod
    def get_L1_loss(cls, a, b):   
        return cls.L1(a, b)

    @classmethod
    def get_L2_loss(cls, a, b):
        return cls.L2(a, b)

    def get_L1_loss_with_same_person(a, b, same_person, batch_size):
        return torch.sum(torch.mean(torch.abs(a - b).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

    def get_L2_loss_with_same_person(a, b, same_person, batch_size):
        return torch.sum(0.5 * torch.mean(torch.pow(a - b, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

    def get_attr_loss(a, b, batch_size):
        L_attr = 0
        for i in range(len(a)):
            L_attr += torch.mean(torch.pow((a[i] - b[i]), 2).reshape(batch_size, -1), dim=1).mean()
        L_attr /= 2.0

        return L_attr

    def hinge_loss(X, positive=True):
        if positive:
            return torch.relu(1-X).mean()
        else:
            return torch.relu(X+1).mean()

    @classmethod
    def get_hinge_loss(cls, Di, label):
        L_adv = 0
        for di in Di:
            L_adv += cls.hinge_loss(di[0], label)
        return L_adv

    def get_BCE_loss(logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    def get_r1_reg(d_out, x_in):
        # zero-centered gradient penalty for real images
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg

    def get_adversarial_loss(logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss
