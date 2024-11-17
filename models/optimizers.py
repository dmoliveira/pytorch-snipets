import torch
from enum import Enum

class OptimizerType(str, Enum):
    SGD="sgd"
    AdamW="adamw"

class Optimizer():

    def __init__(self, model, optimizer_type: OptimizerType, learning_rate: float=0.001):
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        if optimizer_type == OptimizerType.AdamW:
            self.optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        else:
           self.optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def set_scheduler(self, step_size: int = -1, T_max: int = -1, gamma: float = -1):
        self.step_size = step_size
        self.T_max = T_max 
        self.gamma = gamma
        if step_size > 0: # step decay
            self.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            self.scheduler_name = "step_decay"
        elif T_max > 0: # cosine annealing
            self.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
            self.scheduler_name = "cosine_annealing"
        elif gamma > 0: # exponential decay
            self.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            self.scheduler_name = "exponential_decay"
        else:
            self.scheduler_name = "no_decay"

    def get_optimizer(self):
        return self.optim
