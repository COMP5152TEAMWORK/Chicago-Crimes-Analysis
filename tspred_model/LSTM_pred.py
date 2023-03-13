import torch
from LSTM import LSTM
import pandas as pd
import matplotlib.pyplot as plt

file_2022_new = pd.read_csv("../dataset/testset.csv", delimiter=',').to_numpy()
file_train= pd.read_csv("../dataset/trainset.csv", delimiter=',').to_numpy()
max_num = max(file_train[:, -1])
min_num = min(file_train[:, -1])
pred_series=file_2022_new[-7:,-1]

nor_col = []
for i in range(len(pred_series)):
    nor_col.append((pred_series[i] - min_num) / (max_num - min_num))


path_state_dict = "model_state_dict.pkl"
state_dict_load = torch.load(path_state_dict)
device = torch.device("cuda")

pred_outcome=[]
model = LSTM(input_size=1, output_size=1).to(device)
model.load_state_dict(state_dict_load)
model.eval()
model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=device),
                     torch.zeros(1, 1, model.hidden_layer_size, device=device))
for i in range(0,365):
    y_pred_raw = model(torch.tensor(nor_col[i:i + 7], device=device).float())
    nor_col.append(y_pred_raw)
    y_pred=y_pred_raw.item()*(max_num-min_num)+min_num
    pred_outcome.append(y_pred)
    print(f"date:{i+1:3} output:{pred_outcome[i]}")


a = [0,30,58,89,119,150,180,211,242,272,303,333]
labels = ['Jan', 'Feb', 'Mar', 'Apr','May','June','July','Aug','Sep','Oct','Nov','Dec']
plt.xticks(a,labels)
plt.plot(pred_outcome)
plt.show()