import pytorch_lightning as pl
from model import Model
from mnft.callback import NftCallback
from mnft.utils import *
from torch.utils.data import DataLoader
from data import prepare_data
import torch
from mnft.loader import get_train_loader

VERBOSE = False
EPOCHS = 10
BATCH_SIZE = 64
OWNER = "0x34e619ef675d6161868cc16cf929f860f88242f6"

# we have to return datasets!
train, val, test = prepare_data()

train_loader = DataLoader(train, batch_size=BATCH_SIZE)
val_loader = DataLoader(val, batch_size=BATCH_SIZE)
test_loader = DataLoader(test, batch_size=BATCH_SIZE)

get_train_loader(train, BATCH_SIZE, EPOCHS)

model = Model()

trainer = pl.Trainer(
    max_epochs=EPOCHS, callbacks=[NftCallback(OWNER)], enable_progress_bar=VERBOSE
)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# save model
torch.save(model, "test-model")
