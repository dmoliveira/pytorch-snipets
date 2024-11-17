import torch
from time import time

def get_device():
    """Detect and return the appropriate device (CPU, GPU, or MPS)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def difftime(start_time, unit: str="s") -> int:
      """Calculate difference in time in seconds or milliseconds."""
      if unit == "ms":
          return int((time()-start_time)/1000)
      else: # seconds
          return int(time()-start_time)
