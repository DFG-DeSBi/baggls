import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics.classification import MultilabelAccuracy, BinaryAccuracy
from typing import Callable

class SimpleCNN(pl.LightningModule):
    """
        Simple CNN model for binary classification of sequences.
    """
    def __init__(self, seq_len=500, n_classes=1, kernel_size=10, n_fc_units=16, dropout=0.3, lr=1e-3):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=kernel_size),  
            nn.ReLU(),
            nn.BatchNorm1d(8)
        )

        # Calculate the output size of the last convolutional layer
        conv_output_size = self.get_conv_output_size(seq_len)

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, n_fc_units),  # Use calculated conv output size here
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_fc_units, n_fc_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_fc_units, n_classes)
        )

        self.n_classes = n_classes
        self.learning_rate = lr

        self.loss_fn = nn.BCELoss()
        self.accuracy = BinaryAccuracy()

    def get_conv_output_size(self, seq_len):
        dummy_input = torch.zeros(1, 4, seq_len)  
        output = self.conv_layers(dummy_input)
        return int(output.view(1, -1).size(1))  

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_layers(x)
        return torch.sigmoid(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log("train_accuracy", accuracy, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log("test_accuracy", accuracy, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer