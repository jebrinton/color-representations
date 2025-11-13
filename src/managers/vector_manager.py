import torch
import numpy as np

class VectorManager:

    def __init__(self):
        self.vectors = []

    def last(self, tensor: torch.Tensor):
        pass

    def mean(self) -> torch.Tensor:
        """Compute mean of all stored vectors."""
        return torch.stack(self.vectors).mean(dim=0)


    @staticmethod
    def diff_in_means(set_a: torch.Tensor, set_b: torch.Tensor) -> torch.Tensor:
        """
        Compute steering vector as difference between mean activations.
        """
        return set_a.mean(dim=0) - set_b.mean(dim=0)

   
