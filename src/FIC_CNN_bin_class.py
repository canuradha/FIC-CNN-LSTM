#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import gc
import pickle as pkl

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import randn, flatten


#----------------------------Dataset Arrangement----------------------------
#%%
dataset = pd.read_pickle('../Datasets/classified_readings_edited.pkl')
train_set, test_set = train_test_split(dataset, test_size=0.5, random_state=42)

test_train, test_test = train_test_split(train_set, test_size=0.3, random_state=42)
# print(test_test.shape)
train_dataset_full = pd.concat(test_train.to_numpy())
test_dataset_full = pd.concat(test_test.to_numpy())

#%%
# train_dataset_full['type'].shape[0]

def labelMove(df : pd.DataFrame):
    labels = np.zeros(df['type'].shape[0], dtype=np.float64)
    for id,label in enumerate(df['type']):
       if(label != 6.0):
           labels[id] = 1.0
    return labels

move_label = labelMove(train_dataset_full)
move_label_test = labelMove(test_dataset_full)

train_dataset_full['isMovement'] = move_label
test_dataset_full['isMovement'] = move_label_test


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

train_X = train_dataset_full.drop(columns=['type', 'time', 'session', 'isMovement'])
train_y = train_dataset_full['isMovement'].reset_index(drop=True)

test_X = test_dataset_full.drop(columns=['type', 'time', 'session', 'isMovement'])
test_y = test_dataset_full['isMovement'].reset_index(drop=True)

# sampler = RandomUnderSampler(sampling_strategy='not minority', random_state=42)
# X_CNN, y_CNN = sampler.fit_resample(train_X, train_y) 

train_X_sliced = sliding_window(train_X)
test_x_sliced = sliding_window(test_X)
#reshaping
train_X_sliced = np.moveaxis(train_X_sliced, [0,1], [0,-1])
test_x_sliced = np.moveaxis(test_x_sliced, [0,1], [0,-1])

train_y_sliced = np.array(sliding_window(train_y))
test_y_sliced = np.array(sliding_window(test_y))

# y_CNN_converted = train_y_sliced-1
# test_y_converted = test_y_sliced-1

CNN_train = TensorDataset(torch.from_numpy(train_X_sliced).float(), torch.from_numpy(train_y_sliced).float())
CNN_test = TensorDataset(torch.from_numpy(test_x_sliced).float(), torch.from_numpy(test_y_sliced).float())

#%%

batch_size = 40
train_loader = DataLoader(CNN_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(CNN_test, batch_size=batch_size, shuffle=True, drop_last=True)


#%%
class BinCNN(nn.Module):
    def __init__(self):
        super(BinCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=18, kernel_size=6)
        self.conv2 = nn.Conv1d(18, 54 , 6)
        self.conv3 = nn.Conv1d(54,162,6)
        self.conv4 = nn.Conv1d(162, 324, 6)
        self.pool = nn.MaxPool1d(kernel_size = 3,stride=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(batch_size*324*12, 240)
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, batch_size)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.4)
        self.sm = nn.Sigmoid()
        self.norm1 = nn.BatchNorm1d(18)
        self.norm2 = nn.BatchNorm1d(54)
        self.norm3 = nn.BatchNorm1d(162)
        self.norm4 = nn.BatchNorm1d(324)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu(x)
        # x = self.dropout2(x)
        x = self.pool(x)

        x = self.fc1(torch.flatten(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sm(x)
        # x = x.view(20,-1)
        # x2 = x2.view(16, -1)
        # print(x)
        return x

#%%
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
# device = torch.device('cpu')
print(is_cuda)

model = BinCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

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
        labels = labels.float()
        # print(labels.shape)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        # print(preds)
        loss = criterion(preds,labels)
        # print(loss)
        loss.backward()
        # print(labels)
        optimizer.step()
        running_loss += loss
        calAc(preds, labels)
        
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().cpu().numpy())
    train_accuracy.append((accuracy_count/total).detach().cpu().numpy())
    
    print(f'train_loss {train_loss}; accuracy {accuracy_count/total : .3f}')
    # print(f'accuracy_prediction: {accuracy_count/total : .3f}')
    
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
            # labels = labels.float()
            labels = labels.to(device)
            # optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds,labels)
            running_loss += loss
            calAc(preds, labels)
            
    valid_loss = running_loss/len(test_loader)
    valid_losses.append(valid_loss.detach().cpu().numpy())
    valid_accuracy.append((accuracy_count/total).detach().cpu().numpy())
    print(f'valid_loss {valid_loss}; accuracy {accuracy_count/total : .3f}')
    # print(f'accuracy: {accuracy_count/total : .3f}')
    

def calAc(outputs, labels):
    global accuracy_count
    global total
    outLable = torch.round(outputs)
    correctLable = labels
    accuracy_count += (outLable == correctLable).float().sum()
    total += len(correctLable)
    # print(f'accuracy: {}')

#%%
# optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
# criterion = nn.BCELoss()
#%%
#-----------------------Execution----------------------
epochs = 100
for epoch in range(epochs):
    # if epoch == 35:
    #     optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-6)
    print('epochs {}/{}'.format(epoch+1,epochs))
    Train()
    Valid()
    # scheduler.step(np.round(valid_accuracy[-1]*1000)/1000)
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
#----------------------------------Random Tests-----------------------------
cn1 = nn.Conv1d(6,32,6)
cn2 = nn.Conv1d(32, 64, 6)
cn3 = nn.Conv1d(64, 32, 6)
cn4 = nn.Conv1d(32, 64, 6)
m = nn.MaxPool1d(3, stride=1)
fc = nn.Linear(10*32*11, 6)
input1 = randn(10, 6, 40)
output = cn1(input1)
print(input1.shape)
print(output.shape)
output = m(output)
print(output.shape)
output = cn2(output)
output = m(output)
print(output.shape)
output = cn3(output)
output = m(output)
print(output.shape)
output = cn4(output)
output = m(output)
print(output.shape)
# output = flatten(output)
# print(output.shape)
# output = fc(output)
# output.shape
# %%
model.train()
for i, para in enumerate(model.parameters()):
    print(f'{i + 1}th parameter tensor:', para.shape)
    print(para)
    print(para.grad)
# %%
