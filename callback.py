from pytorch_lightning.callbacks import Callback
from utils import hash_training


class NftCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(hash_training(trainer.model, "0x0000dfsdfsd", 44))
