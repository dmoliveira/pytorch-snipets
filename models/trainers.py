import utils 
from time import time

class Trainer():

      def __init__(self, model, early_stopping, loss_function, optimizer):

          self.device = utils.get_device()
          self.model = model
          model.to(self.device)

          self.early_stopping = early_stopping
          self.loss_function = loss_function
          self.optimizer = optimizer
          self.loss_history = []
          self.time_epoch = [] 
          self.epoch = 0

      def reset(self):
          self.loss_history = []
          self.optimizer.zero_grad()

      def train(self, dataloader, is_debug: bool=False):
          start_time_training = time()
          self.epoch += 1
          size = len(dataloader.dataset)
          for batch, (X, y) in enumerate(dataloader):
              _, loss = self.forward(X, y)
              self.backward(loss)

              loss, current = loss.item(), batch * len(X)
              self.loss_history.append(loss)
              if is_debug and current % 1000 == 0:
                  print(f"  • Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
          self.time_epoch.append(utils.difftime(start_time_training))

      def forward(self, X, y):
          _X, _y = X.to(self.device), y.to(self.device)
          pred = self.model(_X)
          loss = self.loss_function(pred, _y)
          return pred, loss

      def backward(self, loss):
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
