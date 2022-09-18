import numpy as np
import hashlib
import os
import torch

SAVE_DIR = "proof"

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)


def create_sequences(batch_size, dataset_size, epochs):
    # create a sequence of data indices used for training
    sequence = np.concatenate(
        [
            np.random.default_rng().choice(
                dataset_size, size=dataset_size, replace=False
            )
            for _ in range(epochs)
        ]
    )
    num_batch = int(len(sequence) // batch_size)
    return np.reshape(sequence[: num_batch * batch_size], [num_batch, batch_size])


def get_train_loader(dataset, batch_size, epochs):
    sequences = create_sequences(batch_size, len(dataset), epochs)

    subset = torch.utils.data.Subset(dataset, np.reshape(sequences, -1))
    print(subset)

    trainloader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, num_workers=0, pin_memory=True
    )

    m = hashlib.sha256()
    for i, d in enumerate(subset.dataset.data):

        ####################
        # TODO: remove for prod
        if i == 100:
            break
        ####################

        print(i, len(subset.dataset.data))
        m.update(d.__str__().encode("utf-8"))

    with open(os.path.join(SAVE_DIR, "hash.txt"), "w") as f:
        f.write(m.hexdigest())

    return trainloader
