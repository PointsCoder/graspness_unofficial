import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
# from knn_pytorch import knn_pytorch
# import knn_pytorch

# def knn(ref, query, k=1):
#   """ Compute k nearest neighbors for each query point.
#   """
#   device = ref.device
#   ref = ref.float().to(device)
#   query = query.float().to(device)
#   inds = torch.empty(query.shape[0], k, query.shape[2]).long().to(device)
#   knn_pytorch.knn(ref, query, inds)
#   return inds

def knn(ref: torch.Tensor, query: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Compute the k-nearest neighbors of each query point from a reference set.

    Args:
        ref (torch.Tensor):   Shape (B, D, N). Reference points.
        query (torch.Tensor): Shape (B, D, M). Query points.
        k (int):              Number of neighbors to retrieve.

    Returns:
        torch.Tensor: Indices of the k nearest neighbors in `ref` for each query point.
                      Shape (B, k, M).
    """
    # Ensure same batch size and dimension
    B, D, N = ref.shape
    B2, D2, M = query.shape
    assert B == B2 and D == D2, (
        f"Incompatible shapes: ref is {ref.shape}, query is {query.shape}"
    )

    # Move both tensors to the same device/dtype if needed
    device = ref.device
    ref = ref.float().to(device)
    query = query.float().to(device)

    # Step 1: Compute pairwise squared distances
    #   distances[b, i, j] = || query[b,:,i] - ref[b,:,j] ||^2
    #   => 'query.unsqueeze(3)' has shape (B, D, M, 1)
    #   => 'ref.unsqueeze(2)'   has shape (B, D, 1, N)
    #   => the subtraction broadcast leads to (B, D, M, N), then we sum over D
    distances = (query.unsqueeze(3) - ref.unsqueeze(2)).pow(2).sum(dim=1)
    # distances has shape (B, M, N)

    # Step 2: Get the indices of the k smallest distances for each query
    # largest=False => we want smallest values
    # indices shape => (B, M, k)
    _, idx = distances.topk(k, dim=2, largest=False)

    # By default we return shape (B, M, k). If you want (B, k, M), transpose:
    idx = idx.transpose(1, 2)
    # idx shape => (B, k, M)

    return idx
