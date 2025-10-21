import torch


def spectral_penalty(W: torch.Tensor, iters=10, kappa=0.8):
    """Estimates the spectral norm using power iteration (as in https://arxiv.org/abs/1802.05957)

    Computes the penalty max(0, spec - kappa)^2
    """
    assert iters > 0
    u = torch.rand(W.size(0), device=W.device)
    u = u / (torch.linalg.vector_norm(u) + 1e-8)
    for _ in range(iters):
        v = W.T @ u
        v = v / (torch.linalg.vector_norm(v) + 1e-8)
        u = W @ v
        u = u / (torch.linalg.vector_norm(u) + 1e-8)
    spec_norm = u @ W @ v  # type:ignore
    return torch.relu(spec_norm - kappa).pow(2), spec_norm
