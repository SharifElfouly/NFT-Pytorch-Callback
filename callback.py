from pytorch_lightning.callbacks import Callback
from utils import hash_training


class NftCallback(Callback):
    def __init__(self):
        self.epochs = 0
        self.hashes = []

    def on_train_epoch_end(self, trainer, pl_module):
        # TODO: get accuracy somehow

        h = hash_training(trainer.model, "0x0000dfsdfsd", 0, self.epochs)
        self.hashes.append(h)

        self.epochs += 1

        print(trainer.val_dataloaders)
        print("3333333")

    def on_train_end(self, trainer, pl_module):
        print("train ended")
        print(self.hashes)
