from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_mnist(is_train: bool=True, batch_size: int=64, shuffle: bool=True):
      """Download MNIST data return a DataLoader."""
      raw_data = datasets.MNIST(root=".", train=is_train, download=True, transform=ToTensor())
      data = DataLoader(raw_data, batch_size=batch_size, shuffle=shuffle)
      return data

def load_data(batch_size: int=64):
    """return training and test Dataloader for MNIST dataset."""
    training_data = get_mnist(is_train=True, batch_size=batch_size)
    test_data = get_mnist(is_train=False, batch_size=batch_size)
    return training_data, test_data
