#!/usr/bin/env python3
import utils
import pandas as pd
import torch
import plotly.graph_objects as go
from time import time 
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -- Utils ----------
def difftime(start_time, unit: str="s") -> int:
    """Calculate difference in time in seconds or milliseconds."""
    if unit == "ms":
        return int((time()-start_time)/1000)
    else: # seconds
        return int(time()-start_time)

def plot_loss(loss_history: list[float]):
    """Display a line chart with the model loss over the iteration/epochs."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_history, mode='lines', name='Training Loss'))
    fig.update_layout(
        title="Training Loss Over Batches",
        xaxis_title="Batch Number",
        yaxis_title="Loss" )
    fig.show()

def plot_metrics(results: dict):
    """Display main model metrics for test set."""
    fig = go.Figure(data=[
        go.Bar(name='Metrics', x=list(results.keys()), y=list(results.values())) ])
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis_tickformat='.1f' )
    fig.show()

# -- Data -----------

def get_mnist(is_train: bool=True, batch_size: int=64, shuffle: bool=True):
    raw_data = datasets.MNIST(root=".", train=is_train, download=True, transform=ToTensor())
    data = DataLoader(raw_data, batch_size=batch_size, shuffle=shuffle) 
    return data

def load_data(batch_size: int=64):
    training_data = get_mnist(is_train=True, batch_size=batch_size)
    test_data = get_mnist(is_train=False, batch_size=batch_size)
    return training_data, test_data

# -- Model ----------

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class EarlyStopping:
    def __init__(self, enabled=False, patience=5, delta=0.01):
        self.enabled = enabled
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        # Check if the validation loss has improved
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
            # Save the best model
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.epochs_no_improve += 1
            # Check if we need to stop
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True

class Trainer():

    def __init__(self, model, early_stopping, loss_function, optimizer):

        self.device = utils.get_device()
        self.model = model
        model.to(self.device)

        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.loss_history = []
        self.epoch = 0

    def reset(self):
        self.loss_history = []
        self.optimizer.zero_grad()

    def train(self, dataloader, is_debug: bool=False):
        self.epoch += 1
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            _, loss = self.forward(X, y)
            self.backward(loss)

            loss, current = loss.item(), batch * len(X)
            self.loss_history.append(loss)
            if is_debug and current % 1000 == 0:
                print(f"  • Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        #plot_loss(loss_history)

    def forward(self, X, y):
        _X, _y = X.to(self.device), y.to(self.device)
        pred = self.model(_X)
        loss = self.loss_function(pred, _y)
        return pred, loss

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

from abc import ABC, abstractmethod

class Evaluator():
    def __init__(self):
        pass

    @abstractmethod
    def predict(self):
        """Calculate predictions, labels and losses."""
        pass

    @abstractmethod
    def calculate_metrics(self):
        """Calculate all metrics related with your problem type (e.g., classification, prediction, etc)."""
        pass

    def eval(self, dataloader, model):
        """Execute the proper model evaluation."""
        # predict and calculate metrics
        all_preds, all_labels, avg_loss = self.predict(dataloader, model)
        results = self.calculate_metrics(all_preds, all_labels, avg_loss)
        # Display results in a table format
        print(pd.DataFrame(results, index=["Metrics"]).T)
        return results

class EvaluatorClassification(Evaluator):

    def __init__(self):
        super().__init__()
        self.accuracy = [] 
        self.precision = [] 
        self.recall = [] 
        self.f1 = [] 
        self.avg_loss = [] 

    def predict(self, dataloader, trainer):
        trainer.model.eval()
        all_preds = []
        all_labels = []
        avg_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                pred, loss = trainer.forward(X, y)
                avg_loss += loss.item()
                all_preds.extend(pred.argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
            avg_loss = avg_loss / len(dataloader)
        return all_preds, all_labels, avg_loss

    def calculate_metrics(self, all_labels, all_preds, avg_loss):
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        self.accuracy.append(accuracy)
        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)
        self.avg_loss.append(avg_loss)
    
        # Prepare results in a dictionary
        results = {
            "Accuracy": accuracy * 100,  # percentage
            "Precision": precision * 100,
            "Recall": recall * 100,
            "F1 Score": f1 * 100,
            "Avg Loss": avg_loss }
        return results
    
def main():

    start_time_total = time()

    # Parameters
    batch_size = 64
    epochs = 1
    early_stopping = EarlyStopping(enabled=False, patience=3, delta=0.01)
    
    print("\nTRAIN-NN Pytorch Script")
    print("\nParameters")
    print(f"  • Batch-Size: {batch_size}")
    print(f"  • Epochs: {epochs}")
    print(f"  • Early-Stopping: ")
    print(f"        • enabled={early_stopping.enabled}")
    print(f"        • patience={early_stopping.patience}")
    print(f"        • delta={early_stopping.delta}")
    print()

    # Load data
    print("01. Load Data")
    start_time_load_data = time()
    training_data, test_data = load_data(batch_size=batch_size)
    print(f"  Loaded in {difftime(start_time_load_data)} s")

    # Create the model
    model = NeuralNetwork()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    trainer = Trainer(model, early_stopping, loss_function, optimizer)
    evaluator = EvaluatorClassification()
    results = None

    print("\n02. Model Training")
    total_time_training = 0 
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        start_time_training = time()
        trainer.train(training_data)
        results = evaluator.eval(test_data, trainer)
        time_epoch = difftime(start_time_training)
        print(f"Epoch time {time_epoch} s")
        total_time_training += time_epoch 

    # Display training loss and test metrics charts 
    #plot_loss(trainer.loss_history)
    #plot_metrics(results)

    print(f"\nModel trained in {total_time_training} s")
    print(f"\n--  Finished training in {difftime(start_time_total)} s")


if __name__ == "__main__":
    main()

