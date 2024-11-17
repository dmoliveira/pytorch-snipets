from torch import nn

class NeuralNetworkFlatten(nn.Module):
      def __init__(self, nn_stack):
          super(NeuralNetworkFlatten, self).__init__()
          self.flatten = nn.Flatten()
          self.linear_relu_stack = nn_stack

      def forward(self, x):
          x = self.flatten(x)
          logits = self.linear_relu_stack(x)
          return logits
