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
