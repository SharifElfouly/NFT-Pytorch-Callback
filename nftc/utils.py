from hashlib import sha256
import time

def sha(string): return sha256(string.encode()).hexdigest()

def get_model_size(model):
    "return model size in mb"
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

def hash_train_data(data):
    # TODO
    # hashing the whole data probably does not make sense
    pass

def hash_training(model, owner, loss, epoch):
    model_hash = hash_model_weights(model)
    size       = str(get_model_size(model))
    t          = str(time.time())
    loss        = str(loss)
    epoch      = str(epoch)

    s = model_hash + owner + size + t + loss + epoch
    return sha(s)

def validate(model_hash, model, owner, loss, epoch):
    "validate if `model_hash` is correct"
    return model_hash == hash_training(model, owner, loss, epoch)
