import numpy as np
import torch
import pandas as pd
from LSTM import LSTM
import matplotlib.pyplot as plt

file_2022_new = pd.read_csv("../dataset/testset.csv", delimiter=',').to_numpy()
file_train = pd.read_csv("../dataset/trainset.csv", delimiter=',').to_numpy()
max_num = max(file_train[:, -1])
min_num = min(file_train[:, -1])
first_pred = file_train[-7:, -1]
for i in range(len(first_pred)):
    first_pred[i] = (first_pred[i] - min_num) / (max_num - min_num)
print(first_pred)

nor_col = []
for i in range(len(file_2022_new)):
    nor_col.append((file_2022_new[i, -1] - min_num) / (max_num - min_num))
pre_dataset = np.hstack((first_pred, nor_col))
pre_dataset = pre_dataset.tolist()
# print(pred_dataset)

path_state_dict = "model_state_dict.pkl"
state_dict_load = torch.load(path_state_dict)
device = torch.device("cuda")

data_pred = []
model = LSTM(input_size=1, output_size=1).to(device)
model.load_state_dict(state_dict_load)
model.eval()  # analysis mode

for i in range(1, len(pre_dataset) - 1):
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=device),
                         torch.zeros(1, 1, model.hidden_layer_size, device=device))
    y_pred = model(torch.tensor(pre_dataset[i - 1:i + 6], device=device).float())
    data_pred.append(y_pred.item())
    print(f"epoach:{i:3} output:{data_pred[i - 1]}")
    if (len(pre_dataset) - 7 == len(data_pred)): break

data_pred_new = []
data_raw_new = []
for i in range(len(pre_dataset)):
    data_raw_new.append(pre_dataset[i] * (max_num - min_num) + min_num)

for i in range(len(data_pred)):
    data_pred_new.append(data_pred[i] * (max_num - min_num) + min_num)


a = [0,30,58,89,119,150,180,211,242,272,303,333]
labels = ['Jan', 'Feb', 'Mar', 'Apr','May','June','July','Aug','Sep','Oct','Nov','Dec']
plt.xticks(a,labels)
plt.plot(data_raw_new[:-8], 'g',label='Actual')
plt.plot(data_pred_new[:-1], 'b',label='Predict')
plt.legend(labels=["Acutal","Predict"],loc='best')
plt.show()

