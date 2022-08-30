from pytorch_lightning.callbacks import Callback
from .utils import hash_training


class NftCallback(Callback):
    def __init__(self, owner):
        self.owner = owner
        self.epochs = 0
        self.hashes = []

    def on_train_epoch_end(self, trainer, pl_module):
        print("on_train_epoch_end")

    def on_validation_epoch_end(self, trainer, pl_module):
        print("on_validation_epoch_end")

        h = hash_training(trainer.model, self.owner, 0, self.epochs)
        self.hashes.append({
            "loss": float(trainer.callback_metrics["loss"]),
            "hash": h
        })

        self.epochs += 1

    def on_train_end(self, trainer, pl_module):
        print("on_train_end")
        # get hash of the model with the lowest loss
        lowest_loss = sorted(self.hashes, key=lambda d: d['loss'], reverse=True)[0]["hash"]
        print(lowest_loss["hash"])
        print(lowest_hash)



