from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from model import Model
from callback import NftCallback
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, random_split
from data import prepare_data

EPOCHS = 20

train, val, test = prepare_data()

train_loader, val_loader, test_loader = DataLoader(train, batch_size=64), DataLoader(val, batch_size=64), DataLoader(test, batch_size=64)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(max_epochs=EPOCHS, callbacks=[NftCallback()])
trainer.fit(model=Model(), train_dataloaders=train_loader, val_dataloaders=val_loader)
