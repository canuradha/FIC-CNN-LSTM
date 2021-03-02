#%%
import pandas as pd
import pickle as pkl
import numpy as np

data = pd.read_pickle('./LSTM_input_configured_paper.pkl')

data.shape
# %%
print(data.dropna())
# %%
