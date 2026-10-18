import random

import numpy as np
import torch

import config


def set_seed(seed=config.SEED):
    """
    Cố định seed cho random / numpy / torch để kết quả huấn luyện tái lập được.

    Lưu ý: vẫn có thể lệch nhẹ do thuật toán không tất định trên GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
