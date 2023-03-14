import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size=100, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, num_layers)
        self.hidden_layer_size=hidden_layer_size
        self.ReLU = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(hidden_layer_size, 50)
        self.linear2 = torch.nn.Linear(50, output_size)
        self.hidden_cell = (torch.zeros(num_layers, 1, self.hidden_layer_size),
                            torch.zeros(num_layers, 1, self.hidden_layer_size))  # (num_layers , batch_size,
        # hidden_size)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear1(lstm_out.view(len(input_seq), -1))
        predictions = self.linear2(predictions)
        return predictions[-1]


def create_inout_sequences(input_data, train_sequence_length):
    inout_seq = []
    data_length = len(input_data)
    for i in range(data_length - train_sequence_length):
        train_seq = input_data[i:i + train_sequence_length]
        train_label = input_data[i + train_sequence_length:i + train_sequence_length + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq
