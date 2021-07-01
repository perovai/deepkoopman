import torch.nn as nn


class SpaceTime(nn.Module):
    def __init__(self, opts):
        # TODO: Add things like no. of hidden layers to opts
        pass


class LSTM(nn.Module):
	# This class is largely derived from 
	# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python on 20210701. 
    def __init__(self, input_size=2, hidden_layer_size=100, output_size=2):
    	# param input_size: number of components in input vector
    	# param output_size: number of components in output vector
    	# param hidden_layer_size: number of components in hidden layer
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]