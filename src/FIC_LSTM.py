#%%
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# LSTM_data_raw['is_bite']
# %%
LSTM_data_raw = pd.read_pickle('LSTM_dataset_paper.pkl')

bite_sequence = LSTM_data_raw['is_bite']
bite_change = np.zeros(1)
value_flipped = 0.0
for index, value in enumerate(bite_sequence):
    if (value != value_flipped):
        bite_change = np.append(bite_change, index)
        value_flipped = value

SV_data = LSTM_data_raw.drop(columns = ['session', 'time'])

#%%
bite_series = pd.Series([SV_data.loc[bite_change[0]: bite_change[1]-1].to_numpy()])

# %%
start = bite_change[1]
for index, end in enumerate(bite_change[2:]):
    bite_series = bite_series.append(pd.Series([SV_data.loc[start: end-1].to_numpy()], index = [index + 1]))
    start = end

bite_series = bite_series.append(pd.Series([SV_data.loc[start: ].to_numpy()], index = [bite_series.shape[0]]))
# %%
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


# %%
with open('LSTM_input_configured_paper.pkl', 'wb') as handle:
    pkl.dump(LSTM_input, handle, protocol=-1)

#%%
print(LSTM_input.shape)
# %%
LSTM_input = pkl.load(open('LSTM_input_configured_paper.pkl', 'rb'))

LSTM_labels = np.array(LSTM_input[:,:, 10], dtype=np.float64) 
LSTM_in = LSTM_input[:,:,:10].astype(np.float)
LSTM_labels = np.max(LSTM_labels,1)

#%%
print(LSTM_in[1])

# %%
X_train, X_test, y_train, y_test = train_test_split(LSTM_in, LSTM_labels, test_size=0.2, random_state=42)

#%%
tensor_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
tensor_test = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

batch_size = 20

train_loader = DataLoader(tensor_train, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(tensor_test, batch_size=batch_size, drop_last=True)

#%%
print(y_train.shape)
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


    def forward(self, input_seq, hidden):
        lstm_out, hidden = self.lstm(input_seq, hidden)
        predictions = self.fc(lstm_out)
        out = self.sigmoid(predictions)
        # out = out[:,-1]
        # print(out)
        return out, hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_layer_size).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_layer_size).zero_().to(device))
        return tuple([e.data for e in hidden])

#%%
input_size = 10
output_size = 1
hidden_dim = 128
layers = 2

model = LSTM(input_size,hidden_dim, output_size, layers)
model.to(device)

lr = 0.001
loss_func = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
patience, trials = 100, 0
best_acc = 0


# print(model)

epochs = 100
counter = 0
# print_freq = 1000
# batch_size = 32
val_min_loss = np.inf

model.train()
for i in range(epochs):
    hidden = model.init_hidden(batch_size)

    for count, [inputs, labels] in enumerate(train_loader):
        model.train()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, h = model(inputs, hidden)
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
        val_loss = loss_func(torch.round(out).squeeze(), y_val)
        total += y_val.size(0)
        correct += (torch.round(out).squeeze() == y_val).sum().item()
        losses.append(val_loss.item())
        # print(out.squeeze())
    
    acc = correct / total

    if i % 5 == 0:
        print(f'Epoch: {i:3d}. Loss: {loss.item():.4f}. Val Loss: {np.mean(losses):.6f} . Acc.: {acc:2.2%}')

    if acc > best_acc:
        trials = 0
        best_acc = acc
        torch.save(model.state_dict(), 'best.pth')
        print(f'Epoch {i} best model saved with accuracy: {best_acc:2.2%}')
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {i}')
            break

# %%

print(val_min_loss)
# %%
