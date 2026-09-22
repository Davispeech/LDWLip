import time
import chardet
import torch


def time_sync():
    """PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}



def get_filetype(filepath):
    with open(filepath, 'rb') as file:
        rawdata = file.read(10000)
        result = chardet.detect(rawdata)
    file.close()
    return result['encoding']



