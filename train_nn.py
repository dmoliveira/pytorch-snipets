#!/usr/bin/env python3
# Local imports
import utils
import data
import evaluation
import plot
from models.early_stopping import EarlyStopping 
from models.trainers import Trainer
from models.optimizers import Optimizer, OptimizerType
from models.neural_networks import NeuralNetworkFlatten

# External imports
import torch
from time import time 
from torch import nn

# -- Model ----------

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def main():

    start_time_total = time()

    # Parameters
    batch_size = 64
    epochs = 10
    early_stopping = EarlyStopping(enabled=False, patience=3, delta=0.01)

    # Create the model
    nn_stack = nn.Sequential(
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )
    model = NeuralNetworkFlatten(nn_stack)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Optimizer(model, OptimizerType.SGD, learning_rate=0.001)
    optimizer.set_scheduler()
    trainer = Trainer(model, early_stopping, loss_function, optimizer.get_optimizer())
    evaluator = evaluation.EvaluatorClassification()
    results = None

    
    print("\nTRAIN-NN Pytorch Script")
    print("\nParameters")
    print(f"  • Batch-Size: {batch_size}")
    print(f"  • Epochs: {epochs}")
    print(f"  • Early-Stopping: ")
    print(f"        • enabled={early_stopping.enabled}")
    print(f"        • patience={early_stopping.patience}")
    print(f"        • delta={early_stopping.delta}")
    print(f"  • Optimizer: ")
    print(f"        • type={optimizer.optimizer_type}")
    print(f"        • learning_rate={optimizer.learning_rate}")
    print(f"        • scheduler_name={optimizer.scheduler_name}")
    print(f"        • step_size={optimizer.step_size}")
    print(f"        • T_max={optimizer.T_max}")
    print(f"        • gamma={optimizer.gamma}")
    print()

    # Load data
    print("01. Load Data")
    start_time_load_data = time()
    training_data, test_data = data.load_data(batch_size=batch_size)
    print(f"  Loaded in {utils.difftime(start_time_load_data)} s")

    print("\n02. Model Training")
    for epoch in range(epochs):
        trainer.train(training_data)
        results = evaluator.eval(test_data, trainer)

    # Display training loss and test metrics charts 
    #plot.plot_loss(trainer.loss_history)
    #plot.plot_metrics(results)

    print(f"\n--  Finished training in {sum(trainer.time_epoch)} s\n")


if __name__ == "__main__":
    main()

