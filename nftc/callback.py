from pytorch_lightning.callbacks import Callback
from .utils import hash_training
from termcolor import cprint

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
            "epoch": self.epochs, 
            "loss": float(trainer.callback_metrics["loss"]),
            "hash": h
        })

        self.epochs += 1

    def on_train_end(self, trainer, pl_module):
        print("on_train_end")
        # get hash of the model with the lowest loss
        self.print_hashes(self.hashes)

    def on_validation_end(self, trainer, pl_module):
        print("on_val_end")
        pass

    def print_hashes(self, losses):
        print()
        cprint("Summary", "green", attrs=["bold"])
        for loss in losses:
            self.print_hash(loss)

        print()

        cprint("Lowest Loss", "green", attrs=["bold"])
        lowest_loss = sorted(self.hashes, key=lambda d: d['loss'], reverse=False)[0]
        self.print_hash(lowest_loss)
        print()

    def print_hash(self, loss):
        e = loss["epoch"]
        l = '{:.3f}'.format(round(loss["loss"], 3))
        h = loss["hash"]
        print(f"epoch {e}: loss {l} - hash {h}")
