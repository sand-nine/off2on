import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

def compute_srank(matrix,thershold=0.01):
    singular_vals = np.linalg.svd(
        matrix.detach().cpu().numpy(), full_matrices=False, compute_uv=False)
    sum_sing = np.sum(singular_vals)
    accumulated_sum = 0
    k = 0
    for value in singular_vals:
        accumulated_sum += value
        k += 1
        if accumulated_sum >= sum_sing*(1-thershold):
            break
    return k

def weight_l2_norm(network):
    l2_norm = 0.0
    for name, param in network.named_parameters():
        if 'bias' not in name:  # This condition excludes bias terms
            l2_norm += torch.sum(param ** 2)
    
    return l2_norm**0.5