import torch
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    def eval(self, dataloader, trainer):
        """Execute the proper model evaluation."""
        # predict and calculate metrics
        time_epoch = trainer.time_epoch[-1]
        all_preds, all_labels, avg_loss = self.predict(dataloader, trainer)
        results = self.calculate_metrics(all_preds, all_labels, avg_loss, time_epoch)
        # Display results in a table format
        results_str = pd.DataFrame(results, index=["Metrics"]).to_string()
        if trainer.epoch == 1:
            print(results_str)
        else:
            print(results_str.split("\n")[-1])
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

    def calculate_metrics(self, all_labels, all_preds, avg_loss, time_epoch):
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
            "Epoch": len(self.accuracy),
            "Accuracy": round(accuracy * 100, 2),  # percentage
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1Score": round(f1 * 100, 2),
            "AvgLoss": round(avg_loss, 2),
            "TimeEpochSecs": time_epoch}
        return results
