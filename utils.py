import numpy as np
import torch
import random
import os
import sys
import scipy.sparse as sp


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def init_run(log_path, log_name, seed):
    set_seed(seed)
    if not os.path.exists(log_path): os.mkdir(log_path)
    f = open(os.path.join(log_path, f'log_{log_name}.txt'), 'w')
    f = Unbuffered(f)
    sys.stderr = f
    sys.stdout = f

def get_sparse_tensor(mat, device):
    coo = mat.tocoo()
    indexes = np.stack([coo.row, coo.col], axis=0)
    indexes = torch.tensor(indexes, dtype=torch.int64, device=device)
    data = torch.tensor(coo.data, dtype=torch.float32, device=device)
    sp_tensor = torch.sparse.FloatTensor(indexes, data, torch.Size(coo.shape)).coalesce()
    return sp_tensor

def generate_daj_mat(dataset):
    train_array = np.array(dataset.train_array)
    users, items = train_array[:, 0], train_array[:, 1]
    row = np.concatenate([users, items + dataset.n_users], axis=0)
    column = np.concatenate([items + dataset.n_users, users], axis=0)
    adj_mat = sp.coo_matrix((np.ones(row.shape), np.stack([row, column], axis=0)),
                            shape=(dataset.n_users + dataset.n_items, dataset.n_users + dataset.n_items),
                            dtype=np.float32).tocsr()
    return adj_mat

class AverageMeter:
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

def gen_name_string(path, *args, extension=None, **kwargs):
    name_str = '{}_{}'.format(
        '_'.join(args),
        '_'.join([f'{k}={kwargs[k]}' for k in kwargs])
    )
    if extension:
        name_str += f'.{extension}'
    return os.path.join(path, name_str)

def groupby_apply(keys: torch.Tensor, values: torch.Tensor, bins: int = 95, reduction: str = "mean", return_histogram: bool = False):
    if reduction == "mean":
        reduce = torch.mean
    elif reduction == "sum":
        reduce = torch.sum
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")
    uniques, counts = keys.unique(return_counts=True)
    groups = torch.stack([reduce(item) for item in torch.split_with_sizes(values, tuple(counts))])
    reduced = torch.zeros(bins, dtype=values.dtype, device=values.device).scatter(dim=0, index=uniques, src=groups)
    if return_histogram:
        hist = torch.zeros(bins, dtype=torch.long, device=values.device).scatter(dim=0, index=uniques, src=counts)
        return reduced, hist
    else:
        return reduced
