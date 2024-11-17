import torch

def get_device():
    """Detect and return the appropriate device (CPU, GPU, or MPS)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
