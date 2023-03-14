import torch
from LSTM import create_inout_sequences, LSTM
import pandas as pd

file_2020_new = pd.read_csv("../dataset/trainset.csv", delimiter=',').to_numpy()
max_num = max(file_2020_new[:, -1])
min_num = min(file_2020_new[:, -1])

nor_col = []
for i in range(file_2020_new.shape[0]):
    nor_col.append((file_2020_new[i, -1] - min_num) / (max_num - min_num))


recorded_MSE = []
device = torch.device("cuda")

epochs_MSE = []
data_pred=[]
data_pred_new=[]
MSE_sum = 0
train_data = torch.tensor(nor_col[:])
input_sequence = create_inout_sequences(torch.tensor(train_data, device=device).float(), 7)
model = LSTM(input_size=1, output_size=1).to(device)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 800
for i in range(1, epochs + 1):
    for sequence, label in input_sequence:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size,device=device),
                             torch.zeros(1, 1, model.hidden_layer_size,device=device))
        y_pred = model(sequence)
        single_loss = loss(torch.squeeze(y_pred), torch.squeeze(label))
        single_loss.backward()
        optimizer.step()
        MSE=single_loss.item()
        epochs_MSE.append(MSE)
    print(f"epoach:{i:3} loss:{single_loss.item():10.10f}")


path_state_dict = "model_state_dict.pkl"
net_state_dict = model.state_dict()
torch.save(net_state_dict, path_state_dict)

