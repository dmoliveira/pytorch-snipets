#!/usr/bin/env python3
import pandas as pd
import torch
import plotly.graph_objects as go
from time import time 
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DEVICE = torch.device("mps") if torch.backends.mps.is_built() else torce.device("cpu")

# -- Utils ----------
def difftime(start_time, unit: str="s"):
    if unit == "ms":
        return int((time()-start_time)/1000)
    else: # seconds
        return int(time()-start_time)

def plot_loss(loss_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_history, mode='lines', name='Training Loss'))
    fig.update_layout(
        title="Training Loss Over Batches",
        xaxis_title="Batch Number",
        yaxis_title="Loss" )
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

def train(dataloader, model, early_stopping, loss_fn, optimizer):
    loss_history = []
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        loss_history.append(loss)
        if current % 1000 == 0:
            print(f"  • Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    plot_loss(loss_history)

def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Prepare results in a dictionary
    results = {
        "Accuracy": accuracy * 100,  # percentage
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100,
        "Avg Loss": test_loss }

    # Display results in a table format
    results_df = pd.DataFrame(results, index=["Metrics"]).T
    print(results_df)

    # Plot results using Plotly
    fig = go.Figure(data=[
        go.Bar(name='Metrics', x=list(results.keys()), y=list(results.values())) ])
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis_tickformat='.1f' )
    fig.show()

def main():

    start_time_total = time()

    # Parameters
    batch_size = 64
    early_stopping = EarlyStopping(enabled=False, patience=3, delta=0.01)
    enabled = early_stopping.enabled
    patience = early_stopping.patience
    delta = early_stopping.delta
    
    print("\nTRAIN-NN Pytorch Script")
    print("\nParameters")
    print(f"  • Batch-Size: {batch_size}")
    print(f"  • Early-Stopping: enabled={enabled} patience={patience} delta={delta}")
    print()

    # Load data
    print("01/. Load Data")
    start_time_load_data = time()
    training_data, test_data = load_data(batch_size=batch_size)
    print(f"  Loaded in {difftime(start_time_load_data)} s")

    # Create the model
    model = NeuralNetwork()
    model.to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    print("\n02/. Model Training")
    start_time_training = time()
    train(training_data, model, early_stopping, loss_function, optimizer)
    print(f"  Model trained in {difftime(start_time_training)} s")
    
    print("\n03/03. Model Testing")
    start_time_test = time()
    test(test_data, model, loss_function)
    print(f"  Model testing in {difftime(start_time_test)} s")

    print(f"\n--  Finished training in {difftime(start_time_total)} s")


if __name__ == "__main__":
    main()

