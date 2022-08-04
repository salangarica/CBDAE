import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def random_seq_contrastive_loss(h, augmentation_factor=1, tau=1, mode='combined'):
    '''
    NCE LOSS calculation
    mode = 'seq' -> Only Consecutive elements (Zc)
    mode = 'random' -> Only Random Selected elements (Zr)
    mode = 'combined' -> Combined elements (Zc + Zr)
    '''
    h = h.squeeze()
    batch_size = h.shape[0]

    # MASK
    mask = np.eye(augmentation_factor*batch_size)
    if mode == 'seq' or mode == 'combined':
        l_past = np.eye(augmentation_factor*batch_size, k=1)
        l_future = np.eye(augmentation_factor*batch_size, k=-1)
        mask_past = 1 - (mask + l_past)
        mask_future = 1 - (mask + l_future)
        mask_past = torch.from_numpy(mask_past).to(device).bool()
        mask_future = torch.from_numpy(mask_future).to(device).bool()
    mask = 1 - mask
    mask = torch.from_numpy(mask).bool().to(device)

    # Cosine similarity
    h_norm = h.norm(dim=1)[:, None]
    h1_norm = h / h_norm
    h2_norm = h / h_norm
    similarities = torch.mm(h1_norm, h2_norm.transpose(0, 1))

    # Loss
    def loss(i,j, mask):
        l_ij = - torch.log(torch.exp(similarities[i, j]/tau)/(torch.sum(mask[i, :]*torch.exp(similarities[i, :]/tau))))
        return l_ij

    final_loss = 0
    if mode == 'random':
        n = batch_size - 1
        p = 2
        for i in range(0, n, p):
            final_loss += loss(i, i + 1, mask) + loss(i + 1, i, mask)
    elif mode == 'seq':
        n = batch_size - 1
        p = 1
        for i in range(0, n, p):
            final_loss += loss(i, i + 1, mask_future) + loss(i + 1, i, mask_past)
    else: #mode == 'combined'
        n = int(batch_size/2) - 1
        p = 1
        for i in range(0, n, p):
            final_loss += loss(i, i + 1, mask_future) + loss(i + 1, i, mask_past)
    final_loss = final_loss/(2*(n + 1)/p)
    return final_loss
