import abc
import torch
from torch.utils.data import DataLoader
from lib.dataset import FaceDataset


class FaceSwapInterface(metaclass=abc.ABCMeta):
    def __init__(self, args, gpu):
        """
        When overrided, super call is required.
        """
        self.args = args
        self.gpu = gpu

    def load_next_batch(self):
        try:
            I_source, I_target, same_person = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            I_source, I_target, same_person = next(self.iterator)
            
        I_source, I_target, same_person = I_source.to(self.gpu), I_target.to(self.gpu), same_person.to(self.gpu)
        return I_source, I_target, same_person

    def set_dataset(self):
        self.dataset = FaceDataset(self.args.dataset_list, same_prob=self.args.same_prob)

    def set_data_iterator(self):
        """
        Construct sampler according to number of GPUs it is utilizing.
        Using self.dataset and sampler, construct dataloader.
        Store Iterator from dataloader as a member variable.
        """
        sampler = torch.utils.data.distributed.DistributedSampler(self.dataset) if self.args.use_mGPU else None
        self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size, pin_memory=True, sampler=sampler, num_workers=8, drop_last=True)
        self.iterator = iter(self.dataloader)

    @abc.abstractmethod
    def initialize_models(self):
        """
        Construct models, send it to GPU, and set training mode.
        Models should be assigned to member variables.

        eg. self.D = Discriminator(input_nc=3).cuda(self.gpu).train() 
        """
        pass

    @abc.abstractmethod
    def set_multi_GPU(self):
        pass

    @abc.abstractmethod
    def load_checkpoint(self):
        """
        Load pretrained parameters from checkpoint to the initialized models.
        """
        pass

    @abc.abstractmethod
    def set_optimizers(self):
        pass

    @abc.abstractmethod
    def set_loss_collector(self):
        """
        Set self.loss_collector as an implementation of lib.loss.LossInterface.
        """
        pass

    @abc.abstractmethod
    @property
    def loss_collector(self):
        """
        loss_collector should be an implementation of lib.loss.LossInterface.
        This property should be assigned in self.set_loss_collector.
        """
        pass

    @abc.abstractmethod
    def train_step(self):
        """
        Implement a single iteration of training. This will be called repeatedly in a loop. 
        This method should return list of images that was created during training.
        Returned images are passed to self.save_image and self.save_image is called in the 
        training loop preiodically.
        """
        pass

    @abc.abstractmethod
    def save_image(self):
        pass

    @abc.abstractmethod
    def save_checkpoint(self):
        pass   
    