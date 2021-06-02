#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
import gc
import pickle as pkl

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# import torch.nn.functional as F
from torch import randn, flatten

#%%
class MicorMCNN(nn.Module):
    def __init__(self):
        super(MicorMCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=6)
        self.conv2 = nn.Conv1d(64, 128 , 6)
        # self.conv3 = nn.Conv1d(64, 128, 2)
        self.pool = nn.MaxPool1d(kernel_size = 3,stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(20*128*5, 120)
        # self.fc2 = nn.Linear(576, 192)
        self.dropout = nn.Dropout(p=0.7)
        self.sm = nn.Sigmoid()
        self.norm = nn.BatchNorm1d(32)
        self.norm2 = nn.BatchNorm1d(128)
        # self.conv4 = nn.Conv1d(128,6,1)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.conv1(x)
        # x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        # x = self.conv3(x)
        # x = self.dropout(x)
        x = self.fc1(torch.flatten(x))
        # x2 = self.fc1(torch.flatten(x))
        # x = self.fc2(x)
        x = self.sm(x)
        x = x.view(20,-1)
        # x2 = x2.view(16, -1)
        # print(x)
        return x
#%%

dataset = pd.read_pickle('../Datasets/classified_readings_edited.pkl')
# dataset[0].head()
train_set, test_set = train_test_split(dataset, test_size=0.5, random_state=42)

test_train, test_test = train_test_split(train_set, test_size=0.3, random_state=42)
# print(test_test.shape)
train_dataset_full = pd.concat(test_train.to_numpy())
test_dataset_full = pd.concat(test_test.to_numpy())

#%%
sample_rate = 100
window = 0.4   # window length (seconds)
step = 0.2  # (seconds)

def makeround(dfr: pd.DataFrame, value):
    remainder = dfr.shape[0] % value
    if(remainder != 0):
        if type(dfr) == pd.DataFrame:
            dfr = dfr.append(pd.DataFrame(np.zeros([value - remainder, dfr.shape[1]]), columns=dfr.columns))
        else:
            dfr = dfr.append(pd.Series(np.zeros(value - remainder)))
        # dfr = pd.append(dfr, np.zeros([value - remainder, dfr.shape[1]]), axis=0)
    return dfr
    

def sliding_window(axis_reading):
    axis_reading = makeround(axis_reading, int(window*sample_rate)) 
    # print(axis_reading.shape)
    index_arr = np.arange(0,axis_reading.shape[0] -  int(step * sample_rate), int(step * sample_rate))
    slides = []
    
    for idx,index in enumerate(index_arr):
        slide = axis_reading.iloc[index: index + int(window*sample_rate)]
        # slide = axis_reading[index: index + int(window*sample_rate)+1]
        if type(axis_reading) == pd.DataFrame:
            slides.append(slide.to_numpy())
        else:
            slides.append(slide.max())
    return slides
#%%
#%%
# train_dataset_full = pd.concat(train_set.to_numpy())
# test_dataset_full = pd.concat(test_set.to_numpy())

train_X = train_dataset_full.drop(columns=['type', 'time', 'session'])
train_y = train_dataset_full['type'].reset_index(drop=True)

test_X = test_dataset_full.drop(columns=['type', 'time', 'session'])
test_y = test_dataset_full['type'].reset_index(drop=True)

sampler = RandomUnderSampler(sampling_strategy='not minority', random_state=42)
# X_CNN, y_CNN = sampler.fit_resample(train_X, train_y) 

X_CNN = train_X
y_CNN = train_y
#%%

train_X_sliced = sliding_window(X_CNN)
test_x_sliced = sliding_window(test_X)
#reshaping
train_X_sliced = np.moveaxis(train_X_sliced, [0,1], [0,-1])
test_x_sliced = np.moveaxis(test_x_sliced, [0,1], [0,-1])
# np.shape(train_X_sliced[0])
#%%
train_y_sliced = np.array(sliding_window(y_CNN))
test_y_sliced = np.array(sliding_window(test_y))

# X_CNN, y_CNN = sampler.fit_resample(train_X_sliced, train_y_sliced)
#%%
# flat_list = [np.array(sublist) for sublist in train_X_sliced for item in sublist]
# np.shape(flat_list)
# flat_list[1]
# np.moveaxis(train_X_sliced, 0,0).shape
# print(np.array(train_X_sliced).shape)
# print(np.shape(train_X_sliced[-1]))
#%%
# train_y.max()
# len(train_y)
def createLablesMatrix(labels):
    mat = np.zeros((len(labels), int(labels.max())))
    for id, value in enumerate(labels):
        mat[id, int(value)-1] = 1.0
    return mat

#%%
# y_CNN.to_numpy()
type(test_x_sliced)
#%%
# test_y_converted = createLablesMatrix(test_y)
# y_CNN_converted = createLablesMatrix(y_CNN)
y_CNN_converted = train_y_sliced-1
test_y_converted = test_y_sliced-1
#%%

CNN_train = TensorDataset(torch.from_numpy(train_X_sliced).float(), torch.from_numpy(y_CNN_converted).float())
CNN_test = TensorDataset(torch.from_numpy(test_x_sliced).float(), torch.from_numpy(test_y_converted).float())



#%%

batch_size = 20
train_loader = DataLoader(CNN_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(CNN_test, batch_size=batch_size, shuffle=True, drop_last=True)

#%%
# train_iter2 = iter(train_loader)
# print(train_iter2.next()[0].unsqueeze(1))
# print(y_CNN_converted.shape)


#%%

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
# device = torch.device('cpu')
print(is_cuda)

model = MicorMCNN().to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

#%%

train_losses = []
valid_losses = []
valid_accuracy= []
train_accuracy = []
accuracy_count = 0.0
total = 0.0
#%%
def Train():
    
    running_loss = .0
    global total
    global accuracy_count
    total = .0
    accuracy_count = .0
    
    model.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.long()
        # print(labels.shape)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        # print(preds)
        loss = criterion(preds,labels)
        loss.backward()
        # print(labels)
        optimizer.step()
        running_loss += loss
        calAc(preds, labels)
        
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().cpu().numpy())
    train_accuracy.append((accuracy_count/total).detach().cpu().numpy())
    
    print(f'train_loss {train_loss}')
    print(f'accuracy_prediction: {accuracy_count/total : .3f}')
    
def Valid():
    running_loss = .0
    global total
    global accuracy_count
    total = .0
    accuracy_count = .0
    
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.long()
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds,labels)
            running_loss += loss
            calAc(preds, labels)
            
    valid_loss = running_loss/len(test_loader)
    valid_losses.append(valid_loss.detach().cpu().numpy())
    valid_accuracy.append((accuracy_count/total).detach().cpu().numpy())
    print(f'valid_loss {valid_loss}')
    print(f'accuracy: {accuracy_count/total : .3f}')
    

def calAc(outputs, labels):
    global accuracy_count
    global total
    outLable = torch.argmax(outputs, dim=1)
    correctLable = labels
    accuracy_count += (outLable == correctLable).float().sum()
    total += len(correctLable)
    # print(f'accuracy: {}')


#%%
#-----------------------Execution----------------------
epochs = 100
for epoch in range(epochs):
    print('epochs {}/{}'.format(epoch+1,epochs))
    Train()
    Valid()
    gc.collect()

#%%
#----------------------------------Plot Model--------------------------------
import matplotlib.pyplot as plt

plt.plot(range(epochs), train_losses[:], label='train loss')
plt.plot(range(epochs), valid_losses[:], label='validation loss')
# plt.plot(range(epochs), valid_accuracy[-80:] , label='validation accuracy')

plt.legend()
plt.show()

#%%

plt.plot(range(epochs), train_accuracy[:], label='train acc')
plt.plot(range(epochs), valid_accuracy[:], label='validation acc')

plt.legend()
plt.show()
#%%
#----------------------Save Model------------------
torch.save(model.state_dict(), 'CNN_best_lr1e_5_bs16_e10_dp60_2.pth')

#%%
# ---------------------Predictions------------------------
model.load_state_dict(torch.load('CNN_best_lr1e_5_bs16_e10_dp60_2.pth'))
model.eval()
batch_size = 16

test_set_full = pd.concat(test_set.to_numpy())
test_set_X = test_set_full.drop(columns=['type', 'time', 'session'])
test_set_y = test_set_full['type'].reset_index(drop=True)

CNN_predict = TensorDataset(torch.from_numpy(test_set_X.to_numpy()).float(), torch.from_numpy(createLablesMatrix(test_set_y)).float())
predict_loader = DataLoader(CNN_predict, batch_size=batch_size, shuffle=False, drop_last=True)

predicted = []
model_values = []
test_losses = []
running_loss = 0.0
with torch.no_grad():
    for idx, (inputs, labels) in enumerate(predict_loader):
        # print(inputs[0])
        inputs = inputs.to(device)
        labels = labels.to(device)
        preds, mod = model(inputs)
        loss = criterion(preds,labels)
        predicted.append(preds.detach().cpu().numpy())
        model_values.append(mod.detach().cpu().numpy())
        running_loss += loss

    predict_loss = running_loss/len(predict_loader)
    test_losses.append(predict_loss.detach().cpu().numpy())
    print(f'predict_loss {predict_loss}')

predicted = np.array(predicted)
model_values = np.array(model_values)

#%%
import matplotlib.pyplot as plt

plt.plot(CNN_predicted.time, CNN_predicted.ft_24, label='predicted')
# plt.plot(range(len(test_set_y)), test_set_y, label='actual')

plt.legend()
plt.show()
#%%
# model_values = np.array(model_values)
print(predicted.shape)
model_values.shape
# test_set_full.shape
# test_set_X.shape
# CNN_predicted['is_bite'].value_counts()
#%%
# -----------------------------Add bite sequence labels------------------
# CNN_predicted = model_values.reshape(-1, 6)
# CNN_predicted = pd.DataFrame(CNN_predicted, columns=np.arange(1,7)).add_prefix('ft_')
CNN_predicted = model_values.reshape(-1, 24)
CNN_predicted = pd.DataFrame(CNN_predicted, columns=np.arange(1,25)).add_prefix('ft_')
CNN_predicted.insert(0,'time',test_set_full.dropna()['time'].values)
CNN_predicted.insert(0, 'session', test_set_full.dropna()['session'].values)
CNN_predicted.insert(len(CNN_predicted.columns), 'is_bite',np.zeros(CNN_predicted.shape[0]))
CNN_predicted.head()
#%%

FIC = pd.read_pickle("../Datasets/FIC.pkl")

df = pd.DataFrame.from_dict(FIC)
bite_ds = pkl.loads(pkl.dumps(df['bite_gt']))
# def labelMovement(session):

#%%B_32_best_l001_e1000_84_38
bite_ds[0]
# %%
def addBiteLabel(session):
    bite_session = bite_ds[int(session['session'].values[0])-1]
    for sequence in bite_session:
        extracted = session[(session['time'] > sequence[0]) & (session['time'] <= sequence[1])]
        extracted.loc[:,'is_bite'] = 1.0
        session[(session['time'] > sequence[0]) & (session['time'] <= sequence[1])] = extracted
    return session

CNN_predicted = CNN_predicted.groupby(by='session').apply(lambda session: addBiteLabel(session))
CNN_predicted['is_bite'].value_counts()

#%%

# CNN_predicted.head()
#%%
# -------------------------- Save Predictions ------------------------
with open('../Datasets/CNN_LSTM_input_24ft.pkl', 'wb') as handle:
    pkl.dump(CNN_predicted, handle, protocol=-1)


#%%
#----------------------------------Random Tests-----------------------------
cn = nn.Conv1d(6,32,20)
m = nn.MaxPool1d(6, stride=4)
fc = nn.Linear(10*32*11, 6)
input1 = randn(10, 6, 40)
output = cn(input1)
print(input1.shape)
print(output.shape)
output = m(output)
print(output.shape)
# output = flatten(output)
# print(output.shape)
# output = fc(output)
# output.shape
# %%
x = [np.random.random((10,10)) for _ in range(5)]
np.shape(x)

#%%


        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)


        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x

#%%
y = np.bincount(train_y)
li = np.nonzero(y)[0]
np.vstack((li,y[li])).T


# %%
