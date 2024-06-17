import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoBranchCNNLSTM(nn.Module):
    def __init__(self, input_channels=2, num_ops=10, hidden_size=64):
        super(TwoBranchCNNLSTM, self).__init__()

        # First branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Compute LSTM input size dynamically based on the output size of the branches
        dummy_input = torch.zeros(1, input_channels, 256, 256)
        out1 = self.branch1(dummy_input)
        out2 = self.branch2(dummy_input)
        lstm_input_size = out1.size(1) * out1.size(2) * out1.size(3) + out2.size(1) * out2.size(2) * out2.size(3)

        # LSTM layer
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, batch_first=True)

        # MLP head
        self.fc = nn.Linear(hidden_size, num_ops)

    def forward(self, x_in, x_out, prev_state=None):
        # First branch
        out1 = self.branch1(x_in)
        out1 = out1.view(out1.size(0), -1)

        # Second branch
        out2 = self.branch2(x_out)
        out2 = out2.view(out2.size(0), -1)

        # Merge branches
        merged = torch.cat((out1, out2), dim=1)

        # LSTM layer
        if prev_state is None:
            # Initialize LSTM state if not provided
            batch_size = x_in.size(0)
            device = x_in.device
            h_0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
            c_0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
            prev_state = (h_0, c_0)
        
        lstm_out, new_state = self.lstm(merged.unsqueeze(1), prev_state)

        # MLP head
        output = self.fc(lstm_out[:, -1, :])  # Taking the last output of LSTM

        return output, new_state



if __name__ == '__main__':
    model = TwoBranchCNNLSTM()
    x = torch.ones((1, 2, 256, 256))
    y = model(x, x, None)
    print(y[0].shape, y[1][0].shape, y[1][0].shape)