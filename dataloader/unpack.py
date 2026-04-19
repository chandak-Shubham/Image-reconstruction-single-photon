import numpy as np

def unpack(arr):

    # take last 128 frames
    arr = arr[-128:]                 # (128, 800, 100, 3)

    # unpack width bits: 100 bytes → 800 bits
    arr = np.unpackbits(arr, axis=2) # (128, 800, 800, 3)

    return arr