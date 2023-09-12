"""Contains Action Recognition LSTM Model Architecture"""
import torch
import torch.nn as nn

class ActionRecognitionLSTM(nn.Module):
    """LSTM for action recognition from keypoints"""
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            hidden_dim: int, 
            dropout: float =0
        ):
        """Initialises model
        Inputs:
            input_dim (int): input dimensions
            output_dim (int): output dimensions
            hidden_dim (int): hidden dimensions in LSTM module
            dropout (float): dropout value for dropout layer
        Outputs:
            None
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(0)


    def forward(self, x: torch.Tensor):
        """Forward call for model
        Inputs:
            x (torch.Tensor): input Tensor with dimension (
                batch_size, max_frames, max_ppl, 17, 2)
        Output:
            
        """ 

        x = x.flatten(start_dim=2)  # batch_size, max_frames, 340
        output = self.lstm(x)[0]
        output = self.dropout(output)
        output = self.relu(output)
        output = self.fc(output)
        output = self.softmax(output)

        return output
