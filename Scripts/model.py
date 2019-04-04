import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

M = 1000
Tx = 50

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()

        self.layer1 = nn.Sequential(
	        nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(7, 11), stride=(2, 2)),
	        nn.ReLU(),
	        nn.MaxPool2d(kernel_size=(2, 2), stride=(1,1)))
        self.layer2 = nn.Sequential(
	        nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(7, 9), stride=(1, 1)),
	        nn.ReLU(), 
	        nn.MaxPool2d(kernel_size=(2, 2), stride=(1,1))        )
     	self.layer3 = nn.Sequential(
	     	nn.Conv2d(in_channels = 4, out_channels=8, kernel_size = (5, 7), stride=(1, 1)), 
	     	nn.ReLU())
     	self.layer4 = nn.Sequential(
			nn.Conv2d(in_channels = 8, out_channels=16, kernel_size = (3, 5), stride=(1, 1)), 
	     	nn.ReLU(),
	     	nn.MaxPool2d(kernel_size=(2, 2), stride=(1,1)))
        self.fc1 = nn.Sequential(
	        nn.Linear(11*16*16, out_features=1400),
	        nn.ReLU()
        )
        self.fc2 = nn.Sequential(
	        nn.Linear(1400, out_features=600),
	        nn.ReLU()
        )
        self.LSTM1 =  nn.LSTM(input_size=600, hidden_size=150, num_layers=1, bidirectional= True)
        self.LSTM2 =  nn.LSTM(input_size=300, hidden_size=75, num_layers=2, bidirectional=True)
        self.LSTM3 =  nn.LSTM(input_size=150, hidden_size=25, num_layers=2, bidirectional=True)
        self.LSTM4 =  nn.LSTM(input_size=50, hidden_size=8, num_layers=3, bidirectional=True)
        self.LSTM5 =  nn.LSTM(input_size=16, hidden_size=2, num_layers=3, bidirectional=True)
        self.LSTM6 =  nn.LSTM(input_size=4, hidden_size=1, num_layers=1)
        for name, param in self.LSTM1.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        for name, param in self.LSTM2.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        for name, param in self.LSTM3.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        for name, param in self.LSTM4.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        for name, param in self.LSTM5.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        for name, param in self.LSTM6.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
    def forward(self, x, Tx, M):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out = out.reshape(out.shape(0), -1)
        out = out.fc1(out)
        out = out.fc2(out)
        input = out.reshape(Tx,M,-1)
        input, _=self.LSTM1(input)
        input, _=self.LSTM2(input)
        input, _=self.LSTM3(input)
        input, _=self.LSTM4(input)
        input, _=self.LSTM5(input)
        input, _=self.LSTM6(input)
        y_out   =nn.Sigmoid(input)
        return y_out       
    
model = ConvModel().to(device)

