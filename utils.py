from hashlib import sha256
import time

def sha(string): return sha256(string.encode()).hexdigest()

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def hash_model_weights(model):
    # we could use `named_parameters()` here as well
    return sha(str(list(model.parameters())))

def hash_data(data):
    # TODO
    # hashing the whole data probably does not make sense
    pass

def hash_training(model, owner, acc):
    model_hash = hash_model_weights(model)
    size = str(get_model_size(model))
    t = str(time.time())
    acc = str(acc)

    s = model_hash + owner + size + t + acc
    return sha(s)