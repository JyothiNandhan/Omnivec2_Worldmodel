"""
LiDAR point cloud patching: FPS + kNN → local patches.
Extracted from Point-BERT, adapted for OmniVec2 batch processing.
"""
import torch


def fps_batch(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Batch Farthest Point Sampling.
    xyz: (B, N, 3) → centroids: (B, n_samples) indices
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, n_samples, dtype=torch.long, device=xyz.device)
    distance  = torch.full((B, N), 1e10, device=xyz.device)
    farthest  = torch.randint(0, N, (B,), device=xyz.device)
    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=xyz.device), farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance, dim=-1)
    return centroids


def make_patches_batch(xyz: torch.Tensor, num_group: int, group_size: int):
    """Create local point patches via FPS + kNN.
    xyz: (B, N, 3) → patches (B, G, K, 3), centers (B, G, 3)
    """
    B, N, _ = xyz.shape
    fps_idx = fps_batch(xyz, num_group)
    centers = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))
    dists   = torch.cdist(centers, xyz)
    _, nn_idx = torch.topk(dists, k=group_size, largest=False)
    nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    patches = torch.gather(
        xyz.unsqueeze(1).expand(-1, num_group, -1, -1), 2, nn_idx_exp)
    patches = patches - centers.unsqueeze(2)
    return patches, centers
