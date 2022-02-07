import abc


class FaceSwapInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def initialize_models(self):
        pass

    @abc.abstractmethod
    def set_dataset(self):
        pass

    @abc.abstractmethod
    def set_dataloader(self):
        pass

    @abc.abstractmethod
    def set_multi_GPU(self):
        pass

    @abc.abstractmethod
    def load_checkpoint(self):
        pass

    @abc.abstractmethod
    def set_optimizers(self):
        pass

    @abc.abstractmethod
    def set_loss_collector(self):
        pass

    @abc.abstractmethod
    def load_next_batch(self):
        pass

    @abc.abstractmethod
    def train_step(self):
        pass

    @abc.abstractmethod
    def save_image(self):
        pass

    @abc.abstractmethod
    def save_checkpoint(self):
        pass   
    