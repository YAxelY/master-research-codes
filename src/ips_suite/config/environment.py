"""Imports, device, seeds, environment helpers"""

import os, sys, math, json, time, random, hashlib, fcntl, contextlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, cohen_kappa_score,
                             average_precision_score, confusion_matrix,
                             balanced_accuracy_score, roc_curve)

SUITE_VERSION = 'v3'
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA = torch.cuda.is_available()


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def get(self, k, default=None):
        return self.__dict__.get(k, default)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DETERMINISTIC = os.environ.get('IPS_DETERMINISTIC', '0').strip() == '1'
if DETERMINISTIC:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def epoch_seed(seed, epoch):
    # a resumed run must draw the same numbers as one that never stopped
    return int(seed) * 1000 + int(epoch)


def environment_info():
    import platform
    where = ('kaggle' if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
             else 'slurm' if 'SLURM_JOB_ID' in os.environ else 'local')
    return {'platform': where, 'host': platform.node(),
            'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version() if CUDA else None,
            'gpu': torch.cuda.get_device_name(0) if CUDA else 'cpu',
            'deterministic': DETERMINISTIC, 'rng': 'per-epoch seeds (v3.1)',
            'code_commit': os.environ.get('IPS_CODE_COMMIT', '').strip() or None}


def _env(key, default=''):
    # an empty variable counts as unset, so the default applies
    return os.environ.get(key, '').strip() or default


def _env_int(key, default):
    v = _env(key)
    return int(v) if v else default


def _env_path(key, default):
    return Path(os.environ.get(key, str(default)))


print(f"IPS suite {SUITE_VERSION} | PyTorch {torch.__version__} | device {device}")
if CUDA:
    print(f"GPU {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
