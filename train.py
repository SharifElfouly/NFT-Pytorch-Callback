import pytorch_lightning as pl
from model import Model
from nftc.callback import NftCallback
from torch.utils.data import TensorDataset, DataLoader, random_split
from data import prepare_data

EPOCHS = 5
BATCH_SIZE = 64
OWNER = "0x34e619ef675d6161868cc16cf929f860f88242f7"

train, val, test = prepare_data()

train_loader = DataLoader(train, batch_size=BATCH_SIZE)
val_loader   = DataLoader(val, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test, batch_size=BATCH_SIZE)

trainer = pl.Trainer(max_epochs=EPOCHS, callbacks=[NftCallback(OWNER)])
trainer.fit(model=Model(), train_dataloaders=train_loader, val_dataloaders=val_loader)
