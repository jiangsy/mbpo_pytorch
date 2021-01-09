import torch
from .actor import Actor
from .critic import QCritic
from .dynamics import RDynamics, EnsembleRDynamics, FastEnsembleRDynamics
from .normalizer import RunningNormalizer, BatchNormalizer

setattr(torch, 'identity', lambda x: x)
setattr(torch, 'swish', lambda x: x * torch.sigmoid(x))
