import abc


class FaceSwapInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def set_models(self):
        pass

    @abc.abstractmethod
    def set_optimizers(self):
        pass

    @abc.abstractmethod
    def set_loss_collector(self):
        pass

    @abc.abstractmethod
    def use_multi_GPU(self):
        pass

    @abc.abstractmethod
    def load_next_batch(self, dataset):
        pass

    @abc.abstractmethod
    def train_step(self, n):
        pass
