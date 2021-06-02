#%%
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# LSTM_data_raw['is_bite']
# %%
# LSTM_data_raw = pd.read_pickle('../Datasets/LSTM_dataset_paper_6_feature.pkl')
LSTM_data_raw = pd.read_pickle('../Datasets/CNN_LSTM_input_24ft.pkl')

#%%
#-----------------------------Slice using sliding Window and make all sessions fixed length-----------------------------------
# sample_rate = 100
# window = 0.2   # window length (seconds)

# def makeFixed(dframe, value):
#     remainder = dframe.shape[0] % value
#     if(remainder != 0):
#         readings = np.zeros([remainder, int(dframe.shape[1] - 3)])
#         timeChange = dframe['time'].iloc[-2] - dframe['time'].iloc[-1]
#         timeframe = np.arange(dframe['time'].iloc[-1], dframe['time'].iloc[-1] + (remainder* timeChange),  timeChange).reshape((remainder, 1))
#         session = np.full((remainder,1), dframe['session'].iloc[0])
#         isbite = np.full((remainder,1), 0)
#         newdf = np.concatenate((session, timeframe, readings, isbite), axis = 1)
#         newdf = pd.DataFrame(newdf, columns=dframe.columns)
#         dframe = pd.concat([dframe, newdf], ignore_index=True)
#     return dframe

# Fixed_LSTM = LSTM_data_raw.groupby(by='session').apply(lambda session: makeFixed(session, int(window*sample_rate)))    

#---------------------------------------------------------------------------------------

#%%
#-------------------------------- Run if No slicing using above ---------------------------
Fixed_LSTM = LSTM_data_raw

#%%
bite_sequence = Fixed_LSTM['is_bite']
bite_change = np.zeros(1)
value_flipped = 0.0
for index, value in enumerate(bite_sequence):
    if (value != value_flipped):
        bite_change = np.append(bite_change, index)
        value_flipped = value

SV_data = Fixed_LSTM.drop(columns = ['session', 'time'])

#%%
bite_series = pd.Series([SV_data.loc[bite_change[0]: bite_change[1]-1].to_numpy()])

# %%
start = bite_change[1]
for index, end in enumerate(bite_change[2:]):
    bite_series = bite_series.append(pd.Series([SV_data.loc[start: end-1].to_numpy()], index = [index + 1]))
    start = end

bite_series = bite_series.append(pd.Series([SV_data.loc[start: ].to_numpy()], index = [bite_series.shape[0]]))


# %%
#-------------------------------reshape bite sequences to fixed size by adding 0 rows front---------------------------------

max_seqence = 0
for bite in bite_series:
    if(bite.shape[0] > max_seqence):
        max_seqence = bite.shape[0] 
# %%
LSTM_input = np.empty((1, max_seqence, bite_series[0].shape[1]))
for bite in bite_series:
    if bite.shape[0] < max_seqence:
        bite = np.concatenate((np.zeros([max_seqence-bite.shape[0], bite.shape[1]]), bite), axis=0)
        LSTM_input = np.append(LSTM_input, [bite], axis = 0)

#---------------------------------------------------------------------------------------------------

#%%
# LSTM_input.shape
# bite_series.shape
max_seqence
# %%
with open('../Datasets/LSTM_CNN_f24_Labelled.pkl', 'wb') as handle:
    pkl.dump(LSTM_input, handle, protocol=-1)

#%%
# print(LSTM_input[3,-10]*1e+3)
# %%
# LSTM_input = pkl.load(open('../Datasets/LSTM_CNN_Labelled.pkl', 'rb'))

LSTM_labels = np.array(LSTM_input[:,:, 24], dtype=np.float64).round(decimals=2)
LSTM_in = LSTM_input[:,:,:24].astype(np.float)
LSTM_labels = np.max(LSTM_labels,1)

#%%
print(LSTM_labels[1])
LSTM_input.shape

# %%
LSTM_test_X, LSTM_rest_X, LSTM_test_y, LSTM_rest_y = train_test_split(LSTM_in, LSTM_labels, test_size=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(LSTM_test_X, LSTM_test_y, test_size=0.3, random_state=42)

#%%
tensor_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
tensor_test = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

batch_size = 20

train_loader = DataLoader(tensor_train, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(tensor_test, batch_size=batch_size, drop_last=True)

#%%
print(X_train.shape)
#%%

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, n_layers):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.4)


    def forward(self, input_seq, hidden):
        hidden_0 = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(input_seq, hidden_0)
        lstm_out = self.dropout(lstm_out)
        predictions = self.fc(lstm_out)
        out = self.sigmoid(predictions)
        out = out[:,-1]
        # print(out)
        return out, hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_layer_size).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_layer_size).zero_().to(device))
        return tuple([e.data for e in hidden])

#%%
input_size = 24
output_size = 1
hidden_dim = 128
layers = 2

model = LSTM(input_size,hidden_dim, output_size, layers)
model.to(device)

lr = 1e-3
loss_func = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
patience, trials = 100, 0
best_acc = 0



#%%

train_losses = []
valid_losses = []

def train():
    running_loss = 0.0
    model.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds, mod = model(inputs,0)
        loss = loss_func(preds.squeeze(),labels.squeeze())
        # print(loss)
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().cpu().numpy())
    
    print(f'train_loss {train_loss}')


def Valid():
    running_loss = .0
    
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds, mod = model(inputs,0)
            loss = loss_func(preds.squeeze(),labels.squeeze())
            running_loss += loss
            
        valid_loss = running_loss/len(test_loader)
        valid_losses.append(valid_loss.detach().cpu().numpy())
        print(f'valid_loss {valid_loss}')

#%%
print(is_cuda)

epochs = 50
counter = 0
# print_freq = 1000
# batch_size = 32
# val_min_loss = np.inf

for i in range(epochs):
    print('epochs {}/{}'.format(i+1,epochs))
    train()
    Valid()
    gc.collect()


#%%
plt.plot(range(40), train_losses[-40:], label='train loss')
plt.plot(range(40), valid_losses[-40:], label='validation loss')

plt.legend()
plt.show()

#%%

for i in range(epochs):
    # hidden = model.init_hidden(batch_size)
    model.train()
    for count, [inputs, labels] in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, h = model(inputs, 0)
        loss = loss_func(outputs.squeeze(),labels)
        loss.backward()
        optimizer.step()
    
        # if((count + 1 )% 6 == 0):
        #     model.eval()
        #     losses = []
        #     vallidate_h = model.init_hidden(batch_size)
        #     for inp, lab in test_loader:
        #         inp, lab = inp.to(device), lab.to(device)
        #         out, val_h = model(inp, vallidate_h)
        #         val_loss = loss_func(out.squeeze(), lab)
        #         losses.append(val_loss.item())
            
        #     model.train()

        #     print("Epoch: {}/{}...".format(i+1, epochs),'\n',
        #           "Step: {}...".format(count + 1),'\n',
        #           "Loss: {:.6f}...".format(loss.item()),'\n',
        #           "Val Loss: {:.6f}".format(np.mean(losses)),'\n')
        #     if np.mean(losses) < val_min_loss:
        #         torch.save(model.state_dict(), './state_dict.pt')
        #         print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_min_loss,np.mean(losses)))
        #         val_min_loss = np.mean(losses)

    model.eval()
    correct, total = 0, 0
    vallidate_h = model.init_hidden(batch_size)
    losses = []
    for x_val, y_val in test_loader:
        x_val, y_val =  x_val.to(device), y_val.to(device)
        out, h = model(x_val, vallidate_h)
        val_loss = loss_func(out.squeeze(), y_val)
        total += y_val.size(0)
        correct += (torch.round(out).squeeze() == y_val).sum().item()
        losses.append(val_loss.item())
        # print(out.squeeze())
    
    acc = correct / total

    if i % 1 == 0:
        print(f'Epoch: {i:3d}. Loss: {loss.item():.4f}. Val Loss: {np.mean(losses):.6f} . Acc.: {acc:2.2%}')

    if acc > best_acc:
        trials = 0
        best_acc = acc
        torch.save(model.state_dict(), 'B_32_best_l001_e1000.pth')
        print(f'Epoch {i} best model saved with accuracy: {best_acc:2.2%}')
    
    #     if trials >= patience:
    #         print(f'Early stopping on epoch {i}')
    #         break

# %%

print(f'Final Model Accuracy: {best_acc:2.2%}')


# %%
model.load_state_dict(torch.load('B_32_best_l001_e1000_84_38.pth'))
model.eval()
correct, total, TP, FN  = 0, 0, 0, 0
batch_size = 32
vallidate_h = model.init_hidden(batch_size)
losses = []
for x_val, y_val in test_loader:
    x_val, y_val =  x_val.to(device), y_val.to(device)
    out, h = model(x_val, vallidate_h)
    val_loss = loss_func(torch.round(out).squeeze(), y_val)
    total += y_val.size(0)
    pred = torch.round(out).squeeze().detach().to('cpu').numpy()
    correct += np.sum(( pred == y_val))
    # print()
    TP += np.sum((pred == y_val and pred==1.0))
    FN += np.sum((pred != y_val and pred==0.0))
    losses.append(val_loss.item())
# %%
TP
# %%
