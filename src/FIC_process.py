#%%
import pickle as pkl
import pandas as pd
import numpy as np 
# from scipy.signal import medfilt
from scipy.ndimage import median_filter

FIC = pd.read_pickle("../Datasets/FIC/FIC.pkl")

df = pd.DataFrame.from_dict(FIC)

#%%
synced_data = pkl.loads(pkl.dumps(df['signals_proc']))
synced_smoothed = pkl.loads(pkl.dumps(df['signals_proc']))
micro_mvmt = pkl.loads(pkl.dumps(df['mm_gt']))
# test[1].shape
#%%
synced_data[1].shape

# -----------------Preporcessing--------------------------------
#%%
# smoothening the triple axis data with a 5th order median filter
for ses_id, session in enumerate(synced_data):
    for read_ax in range(6):
        readings = session[:,read_ax + 1].copy()
        smoothed = median_filter(readings,size = 5)
        synced_smoothed[ses_id][:,read_ax + 1] = smoothed

#%%
# Remove gravity 
for ses_id,session in enumerate(synced_smoothed):
    axel_readings = session[:,1:4].copy()
    gravity = np.array([0,0,0])
    # print(ses_id, '_before: ', axel_readings[1])
    for i, reading in enumerate(axel_readings):
        gravity = 0.9*gravity + 0.1*reading
        # print(ses_id,gravity)
        axel_readings[i] = reading - gravity
    # print(ses_id ,'_after: ', axel_readings[1]) 
    session[:, 1:4] = axel_readings  

# type(synced_smoothed)
#------------------------Feature Extraction-------------------------------
#%%

sample_rate = 100
window = 0.2   # window length (seconds)
step = 0.1  # (seconds)

def makeround(arr, value):
    remainder = arr.shape[0] % value
    if(remainder != 0):
        arr = np.append(arr, np.zeros([value - remainder]), axis=0)
    return arr
    

def feature_ext(axis_reading):
    # remainder = axis_reading.shape[0] % int(window*sample_rate)
    # if(remainder != 0):
    #     axis_reading = np.append(axis_reading, np.zeros([int(window*sample_rate) - remainder]), axis=0)
    axis_reading = makeround(axis_reading, int(window*sample_rate)) 
       
    index_arr = np.arange(0,axis_reading.shape[0] -  int(step * sample_rate), int(step * sample_rate))
    features = np.zeros([index_arr.shape[0], 9],dtype=object)
    
    for idx,index in enumerate(index_arr):
        slide = axis_reading[index: index + int(window*sample_rate)+1]
        fft = np.fft.fft(slide)
        features[idx][:] = [
            (slide[:-1]*slide[1:] < 0).sum() + np.count_nonzero(slide==0),             # zero crossing count 
            np.mean(slide),                                                            # mean
            np.std(slide),                                                             # std
            np.max(slide),                                                             # max value
            np.min(slide),                                                             # min value
            np.ptp(slide),                                                             # range of the values
            np.sum(fft.real*fft.real)/fft.size,                                        # normalized power
            fft.real,                                                                  # discreet fourier transform coefficients
            abs(slide).sum()                                                                # sum of values (for moving average calculations)
        ] 
    return features
#%%
column_names = ['time', 'ax_zero_corss', 'ax__mean', 'ax_std', 'ax_max', 'ax_min', 'ax_range', 'ax_normalized_energy', 'ax_dft_coeff',
                    'ay_zero_corss', 'ay__mean', 'ay_std', 'ay_max', 'ay_min', 'ay_range', 'ay_normalized_energy', 'ay_dft_coeff',
                    'az_zero_corss', 'az__mean', 'az_std', 'az_max', 'az_min', 'az_range', 'az_normalized_energy', 'az_dft_coeff', 
                    'gx_zero_corss', 'gx_mean', 'gx_std', 'gx_max', 'gx_min', 'gx_range', 'gx_normalized_energy', 'gx_dft_coeff',
                    'gy_zero_corss', 'gy__mean', 'gy_std', 'gy_max', 'gy_min', 'gy_range', 'gy_normalized_energy', 'gy_dft_coeff',
                    'gz_zero_corss', 'gz__mean', 'gz_std', 'gz_max', 'gz_min', 'gz_range', 'gz_normalized_energy', 'gz_dft_coeff', 
                    'sma_acc', 'sma_gy']
def feature_ses(row):
    features = row[:,1:]
    extracted = np.apply_along_axis(feature_ext, 0, features)
    mv_avg_ax = (extracted[:,8,:3].sum(axis=1)/int(window*sample_rate))[:, None]
    mv_avg_gy = (extracted[:,8,3:].sum(axis=1)/int(window*sample_rate))[:, None]
    extracted = extracted[:,:8,:]
    extracted = extracted.transpose(0,2,1).reshape(-1,48)
    extracted = np.concatenate((np.concatenate((extracted, mv_avg_ax), axis=1),mv_avg_gy), axis=1)

    times = row[:,0]
    times = makeround(times, int(window*sample_rate))
    index_arr = np.arange(0,times.shape[0] -  int(step * sample_rate), int(step * sample_rate))
    f_times = np.array([times[i] for i in index_arr])[:, None]

    extracted = pd.DataFrame(np.concatenate((f_times, extracted), axis=1), columns=column_names)
    dft = pd.concat([
        pd.DataFrame(extracted['ax_dft_coeff'].values.tolist()).add_prefix('ax_dft_'),
        pd.DataFrame(extracted['ay_dft_coeff'].values.tolist()).add_prefix('ay_dft_'), 
        pd.DataFrame(extracted['az_dft_coeff'].values.tolist()).add_prefix('az_dft_'), 
        pd.DataFrame(extracted['gx_dft_coeff'].values.tolist()).add_prefix('gx_dft_'), 
        pd.DataFrame(extracted['gy_dft_coeff'].values.tolist()).add_prefix('gy_dft_'), 
        pd.DataFrame(extracted['gz_dft_coeff'].values.tolist()).add_prefix('gz_dft_')
    ], axis=1)
    extracted = extracted.drop(columns=['ax_dft_coeff', 'ay_dft_coeff', 'az_dft_coeff', 'gx_dft_coeff', 'gy_dft_coeff', 'gz_dft_coeff'], axis=1)
    extracted = pd.concat([extracted, dft], axis=1)
    return extracted

feat_full = synced_smoothed.apply(feature_ses)

#%%
print(feat_full[0].shape)
#%%
with open('featureset.pkl', 'wb') as handle:
    pkl.dump(feat_full, handle, protocol=-1)


# ----------------------------Labelling Data----------------------------------
#%%
features_imported = pd.read_pickle('featureset.pkl')
#%%
features_classified = pd.Series(features_imported)

for index, session in enumerate(features_imported):
    mvts = pd.DataFrame(micro_mvmt[index], columns=['start', 'end','type'])
    session_mvts = mvts.groupby(by='type').apply(lambda mtype: 
            mtype.apply(
                lambda row: session[(session['time'] > row[0]) & (session['time'] <= row[1])], 
                axis=1
            )
        )
    features_classified[index] = session_mvts.groupby(level=0).apply(lambda group: pd.concat(group.tolist())).reset_index(level=0)
    # print(type(session_mvts[(1.0)]))

#--------------------------------Add Session Column
#%%
rowcount = features_classified.aggregate(lambda a: a.count())['type'].sum()

session_no = np.zeros([rowcount,1])
session_id = 0
for index, session in enumerate(features_classified):  
    session_no[session_id: session_id + session.shape[0]] = int(index +1)
    session.insert(0,'session',session_no[session_id: session_id + session.shape[0]])
    session_id = session_id + int(session.shape[0])
# %%
with open('classified_features.pkl', 'wb') as handle:
    pkl.dump(features_classified, handle, protocol=-1)

# ------------------------------------------------------------------------------------------------------

#%%
import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(synced_data[0][:,0], synced_data[0][:, 1])
ax1.grid()
ax2.plot(synced_smoothed[0][:,0], synced_smoothed[0][:,1])
ax2.grid()

plt.show()
plt.close()
plt.clf()


# %%


# feature_ax = synced_smoothed[0][:,0:2]
# remainder = feature_ax.shape[0] % int(window*sample_rate)

# if(remainder != 0):
#     feature_ax = np.append(feature_ax, np.zeros([int(window*sample_rate) - remainder, 2]), axis=0)


# index_arr = np.arange(0,feature_ax.shape[0] -  int(step * sample_rate), int(step * sample_rate))
# features = np.zeros([index_arr[-1]+1, 5])
# for index in index_arr:
#     slide = feature_ax[index: index + int(window*sample_rate)]
#     features[index][0:2] = [slide[int(step*sample_rate)-1][0], np.mean(slide[:,1])]
#     # print(slide.shape)


# ------------------------------------accessing multi-indexed datframe----------------------------
# svm1 = pd.concat([t.loc[1.0], t.loc[2.0]])
# svm1