import numpy as np
import hashlib
import os
import torch


def create_sequences(batch_size, dataset_size, epochs):
    # create a sequence of data indices used for training
    sequence = np.concatenate(
        [
            np.random.default_rng().choice(
                dataset_size, size=dataset_size, replace=False
            )
            for i in range(epochs)
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
    for d in subset.dataset.data:
        m.update(d.__str__().encode("utf-8"))
    f = open(os.path.join(save_dir, "hash.txt"), "x")
    f.write(m.hexdigest())
    f.close()

    return trainloader
