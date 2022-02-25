import abc
import torch
from torch.utils.data import DataLoader
from lib.dataset import FaceDatasetTrain, FaceDatasetValid


class FaceSwapInterface(metaclass=abc.ABCMeta):
    """
    Base class for face swapping GAN model. This base class can also be used 
    for neural network models with different purposes if some of concrete methods 
    are overrided appropriately. Exceptions will be raised when subclass is being 
    instantiated but abstract methods were not implemented. 
    """

    def __init__(self, args, gpu):
        """
        When overrided, super call is required.
        """
        self.args = args
        self.gpu = gpu

    def load_next_batch(self):
        """
        Load next batch of source image, target image, and boolean values that denote 
        if source and target are identical.
        """
        try:
            I_source, I_target, same_person = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            I_source, I_target, same_person = next(self.train_iterator)
            
        I_source, I_target, same_person = I_source.to(self.gpu), I_target.to(self.gpu), same_person.to(self.gpu)
        return I_source, I_target, same_person

    def set_dataset(self):
        """
        Initialize dataset using the dataset paths specified in the command line arguments.
        """
        self.train_dataset = FaceDatasetTrain(self.args.train_dataset_root_list, self.args.isMaster, same_prob=self.args.same_prob)
        if self.args.valid_dataset_root:
            self.valid_dataset = FaceDatasetValid(self.args.valid_dataset_root, self.args.isMaster)

    def set_data_iterator(self):
        """
        Construct sampler according to number of GPUs it is utilizing.
        Using self.dataset and sampler, construct dataloader.
        Store Iterator from dataloader as a member variable.
        """
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.args.use_mGPU else None
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_per_gpu, pin_memory=True, sampler=sampler, num_workers=8, drop_last=True)
        self.train_iterator = iter(self.train_dataloader)

    def set_valid_set(self):
        """
        Predefine test images only if args.valid_dataset_root is specified.
        These images are anchored for checking the improvement of the model.
        """
        if self.args.valid_dataset_root:
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.args.batch_per_gpu, num_workers=8, drop_last=True)
            I_source, I_target = next(iter(self.valid_dataloader))
            self.valid_source, self.valid_target = I_source.to(self.gpu), I_target.to(self.gpu)

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

    @property
    @abc.abstractmethod
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
    def validation(self):
        """
        Test the model using a predefined valid set.
        This method includes util.save_image and returns nothing.
        """
        pass

    @abc.abstractmethod
    def save_image(self):
        """
        Batch of images from train_step is organized into a table and saved as an image.
        """
        pass

    @abc.abstractmethod
    def save_checkpoint(self):
        """
        Save model and optimizer parameters.
        """
        pass   
    