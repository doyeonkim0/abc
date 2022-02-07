import abc


class FaceSwapInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__():
        """
        self.{args, gpu, model_name} should be initialized.
        Extra member variables can also be initialized here if necessary.
        """
        pass
    
    @abc.abstractmethod
    def initialize_models(self):
        """
        Construct models, send it to GPU, and set training mode.
        Models should be assigned to member variables.

        eg. self.D = Discriminator(input_nc=3).cuda(self.gpu).train() 
        """
        pass

    @abc.abstractmethod
    def set_dataset(self):
        """
        Set dataset as a member variable.

        eg. self.dataset = FaceDataset(dataset_list, same_prob)
        """
        pass

    @abc.abstractmethod
    def set_data_iterator(self):
        """
        Construct sampler according to number of GPUs it is utilizing.
        Using self.dataset and sampler, construct dataloader.
        Store Iterator from dataloader as a member variable.
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
    def load_next_batch(self):
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
    