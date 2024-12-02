import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from generate_landmark_data import label_dict_from_config_file

list_label = label_dict_from_config_file("hand_gesture.yaml")


class CustomImageDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.labels = torch.from_numpy(
            self.data.iloc[:, 0].to_numpy()
        )  # Labels from the first column

    def __len__(self):
        return len(self.data)  # Total number of samples

    def __getitem__(self, idx):
        # Get label and features for the given index
        one_hot_label = self.labels[idx]
        torch_data = torch.from_numpy(
            self.data.iloc[idx, 1:].to_numpy(dtype=np.float32)
        )  # Features as tensor
        return torch_data, one_hot_label


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, len(list_label)),
        )

    def forward(self, x):
        # Forward pass: Flatten input and pass through the network
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

    def predict(self, x, threshold=0.8):
        # Predict with confidence threshold
        logits = self(x)  # Forward pass
        softmax_prob = nn.Softmax(dim=1)(logits)  # Convert logits to probabilities
        chosen_ind = torch.argmax(
            softmax_prob, dim=1
        )  # Select class with highest probability
        # Return class if confidence exceeds threshold, else -1
        return torch.where(softmax_prob[0, chosen_ind] > threshold, chosen_ind, -1)

    def predict_with_known_class(self, x):
        # Predict without threshold, return the class with highest probability
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        return torch.argmax(softmax_prob, dim=1)

    def score(self, logits):
        # Score the logits by returning the negative of the maximum logit value
        return -torch.amax(logits, dim=1)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience  # Number of allowed epochs without improvement
        self.min_delta = min_delta  # Minimum change to qualify as an improvement
        self.counter = 0  # Counter for epochs without improvement
        self.watched_metrics = (
            np.inf
        )  # Best metric value observed so far (initialize to infinity)

    def early_stop(self, current_value):
        # Check if the current metric value improves or not
        if current_value < self.watched_metrics:
            # Update best observed metric and reset counter
            self.watched_metrics = current_value
            self.counter = 0
        elif current_value > (self.watched_metrics + self.min_delta):
            # Increment counter if no significant improvement
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop if patience limit is exceeded
        return False  # Continue training otherwise
