# core/neural_network.py
"""
Neural network components for agent intelligence in the neuro-fuzzy multiagent OS.
Implements basic PyTorch models for self-learning and adaptive behavior.
"""
import torch
import torch.nn as nn


class SimpleNeuralNetwork(nn.Module):
    """
    Basic feedforward neural network for agent learning.
    Can be used as a building block for AI drivers or adaptive agents in the intelligent OS.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network layers.
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of units in the hidden layer.
            output_size (int): Number of output features.
        """
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
