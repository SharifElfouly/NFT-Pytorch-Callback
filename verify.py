from nftc.utils import verify

PATH  = "/Users/shafu/NFT-Pytorch-Callback/xxxx-model"
OWNER = "0x34e619ef675d6161868cc16cf929f860f88242f7"
LOSS  = -0.781
EPOCH = 2
DATE  = "2022-09-05 12:29:05.084334"

HASH = "4535573218e5945ca8c36d8dab34f07b613f11faeb35d5304ede253445107150"

assert verify(HASH, PATH, OWNER, LOSS, EPOCH, DATE)
