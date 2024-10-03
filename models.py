import torch.nn as nn        
import torch

class NNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim:int = 64) -> None:
        """
        Constructor for the NNet class.
        
        :param input_dim: The input dimension
        :type input_dim: int
        :param output_dim: The output dimension
        :type output_dim: int
        :param hidden_dim: The hidden dimension
        :type hidden_dim: int
        """
        
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to forward pass the input data.
        
        :param x: The input data
        :type x: torch.Tensor
        :returns: The output data
        :rtype: torch.Tensor
        """
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    
class Agent(NNet):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim:int = 64) -> None:
        """
        Constructor for the Agent class.
        
        :param input_dim: The input dimension
        :type input_dim: int
        :param output_dim: The output dimension
        :type output_dim: int
        :param hidden_dim: The hidden dimension
        :type hidden_dim: int
        """
        super(Agent, self).__init__(input_dim, output_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.MSELoss()
        
    def train(self, x: torch.Tensor, y: torch.Tensor, epochs:int =100, batch_size: int= 32) -> float:
        """
        Method to train the model.
        
        :param x: The input data
        :type x: torch.Tensor
        :param y: The target data
        :type y: torch.Tensor
        :param epochs: The number of epochs
        :type epochs: int
        :param batch_size: The batch size
        :type batch_size: int
        :returns: The loss
        :rtype: float
        """
        
        for e in range(epochs):
            for i in range(0, len(x), batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+len(x_batch)]
                
                self.optimizer.zero_grad()
                output = self.forward(x_batch)
                loss = self.loss(output, y_batch)
                loss.backward()
                self.optimizer.step()
                
            yield e+1, loss.item()
                
        # return loss.item()
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to predict the output.
        
        :param x: The input data
        :type x: torch.Tensor
        :returns: The output data
        :rtype: torch.Tensor
        """
        
        return self.forward(x)