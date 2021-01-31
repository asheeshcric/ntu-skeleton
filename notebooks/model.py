import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTM(nn.Module):
    def __init__(self, params):
        super(ConvLSTM, self).__init__()
        
        # Here, input_channel = 1 as there are only 3D coordinates of 25 joints
        # So, each frame is provided as input in the following format: (1, 25, 3)
        # Each segment has N frames and the batch size is B_S, so input size is: (B_S, N, 1, 25, 3)
        self.bcc = params.bcc # Base convolution channels (changeable parameter for the model)
        
        self.conv1 = nn.Conv2d(1, self.bcc, kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(self.bcc, 2*self.bcc, kernel_size=(2, 1))
        self.conv3 = nn.Conv2d(2*self.bcc, 4*self.bcc, kernel_size=(2, 1))
        
        self._to_linear, self._to_lstm = None, None
        x = torch.randn(params.BATCH_SIZE, params.seg_size, params.num_channels, params.num_joints, params.num_coord)
        self.convs(x)
        
        self.lstm = nn.LSTM(input_size=self._to_lstm, hidden_size=256, num_layers=1, batch_first=True)
        
        self.fc1 = nn.Linear(256, self.bcc*1)
        self.fc2 = nn.Linear(self.bcc*1, params.num_classes)
        
    def convs(self, x):
        batch_size, timesteps, c, h, w = x.size()
        x = x.view(batch_size*timesteps, c, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self._to_linear is None:
            # Only used for a random first pass: done to know what the output of the convnet is
            self._to_linear = int(x[0].shape[0]*x[0].shape[1]*x[0].shape[2])
            r_in = x.view(batch_size, timesteps, -1)
            self._to_lstm = r_in.shape[2]

        return x
    
    
    def forward(self, x):
        batch_size, timesteps, c, h, w = x.size()
        cnn_out = self.convs(x)
        r_in = cnn_out.view(batch_size, timesteps, -1)
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out = self.fc1(r_out[:, -1, :])
        r_out = self.fc2(r_out)
        return F.log_softmax(r_out, dim=1)