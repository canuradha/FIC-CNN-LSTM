#%%
import pickle as pkl
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from itertools import combinations
# from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import BaggingClassifier

#%%
dataset = pd.read_pickle('classified_features.pkl')
rowcount = dataset.aggregate(lambda a: a.count())['type'].sum()

#%%
session_no = np.zeros([rowcount,1])
session_id = 0
for index, session in enumerate(dataset):  
    session_no[session_id: session_id + session.shape[0]] = int(index +1)
    session.insert(0,'session',session_no[session_id: session_id + session.shape[0]])
    session_id = session_id + int(session.shape[0])

#%%
# train_dataset, test_dataset = train_test_split(dataset, test_size=0.5, random_state=42)

train_svm, test_svm = train_test_split(dataset, test_size=0.7, random_state=42)

# print(train_svm.shape)
train_dataset_full = pd.concat(train_svm.values)

# train_dataset_full['type'].value_counts()
# train_dataset_full.isna()

# %%filename
#  Bagging Classifier
sv = BaggingClassifier(svm.SVC(random_state=42, decision_function_shape='ovo', class_weight='balanced', C=100, gamma=0.1), n_jobs=4)
# sv = svm.SVC(probability=True, class_weight='balanced', random_state=42, C=100, gamma=0.1)

# X_sv = train_dataset_full[train_dataset_full['type'] != 6.0 ].drop(columns=['type', 'session'])
# y_sv = train_dataset_full[train_dataset_full['type'] != 6.0 ]['type']

X_sv = train_dataset_full.drop(columns=['type', 'session'])
y_sv = train_dataset_full['type']

# print(y_sv.isna())
#%%
mod_sv = sv.fit(X_sv, y_sv)
 #%%
# sv.estimators_
#%%
# del train_dataset_full

with open('svm_trained_with_type_6.pkl', 'wb') as handle:
    pkl.dump(mod_sv, handle, protocol=-1)
#%%

#%%
svm_model = mod_sv
# svm_model = pkl.load(open('svm_trained_paper.pkl', 'rb'))
test_svm_full = pd.concat(test_svm.values)
svm_predicted  = svm_model.predict_proba(test_svm_full.drop(columns=['type','session']).dropna())
# t = test_svm_full.dropna()
# t[t.isnull().any(axis=1)]
# svm_predicted
#%%
with open('svm_multi_model.pkl', 'wb') as handle:
    pkl.dump(svm_model, handle, protocol=-1)
# %%
# -----------------------Save Predicted Model-------------------------------
with open('svm_paper_testset_perdictied_proba.pkl', 'wb') as handle:
    pkl.dump(svm_predicted, handle, protocol=-1)

#%%
# t = np.zeros(svm_predicted.shape[0])
# t[:] = 1 - svm_predicted[:,0]
test_svm_full['type'].value_counts()
# svm_predicted[1000: 1010]
#%%
#------------------------Insert time and session ids to the predicted data-------------------
svm_predicted = pd.DataFrame(pd.read_pickle('../Datasets/svm_paper_predicted_with_type_6.pkl'), columns=np.arange(1,7))
# svm_predicted = svm_predicted.add_prefix('sv_')
svm_predicted = pd.DataFrame(svm_predicted, columns=np.arange(1,7)).add_prefix('sv_')
svm_predicted.insert(0,'time',test_svm_full.dropna()['time'].values)
svm_predicted.insert(0, 'session', test_svm_full.dropna()['session'].values)
svm_predicted.insert(len(svm_predicted.columns), 'is_bite',np.zeros(svm_predicted.shape[0]))
svm_predicted.head()

#%%
# print(svm_predicted)
svm_predicted.tail()
# %%
# -----------------------Label the predicted dataset with activities----------------

FIC = pd.read_pickle("../Datasets/FIC/FIC.pkl")

df = pd.DataFrame.from_dict(FIC)
bite_ds = pkl.loads(pkl.dumps(df['bite_gt']))
# def labelMovement(session):

#%%
bite_ds[0]
# %%
def addBiteLabel(session):
    bite_session = bite_ds[int(session['session'].values[0])-1]
    for sequence in bite_session:
        extracted = session[(session['time'] > sequence[0]) & (session['time'] <= sequence[1])]
        extracted.loc[:,'is_bite'] = 1.0
        session[(session['time'] > sequence[0]) & (session['time'] <= sequence[1])] = extracted
    return session

svm_predicted = svm_predicted.groupby(by='session').apply(lambda session: addBiteLabel(session))
svm_predicted['is_bite'].value_counts()
#%%
print(svm_predicted.isna())
# %%
with open('LSTM_dataset_paper.pkl', 'wb') as handle:
    pkl.dump(svm_predicted, handle, protocol=-1)
# %%
# %%
