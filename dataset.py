import numpy as np
from numpy.core.numeric import indices
from scipy import signal
from scipy.signal import decimate, resample_poly
from torch.utils.data import Dataset
import torch
import os
import warnings

# Filtering method

def _norm_freq(frequency=None, sampling_rate=500.):
    try:
        frequency = float(frequency)
    except TypeError:
        # maybe frequency is a list or array
        frequency = np.array(frequency, dtype='float')

    Fs = float(sampling_rate)

    wn = 2. * frequency / Fs

    return wn

def get_filter(ftype='FIR',
               band='lowpass',
               order=None,
               frequency=None,
               sampling_rate=500.):

    frequency = _norm_freq(frequency, sampling_rate)

    b, a = [], []
    
    if order % 2 == 0:
      order += 1
    a = np.array([1])
    if band in ['lowpass', 'bandstop']:
      b = signal.firwin(numtaps=order,
                          cutoff=frequency,
                          pass_zero=True)
    elif band in ['highpass', 'bandpass']:
      b = signal.firwin(numtaps=order,
                          cutoff=frequency,
                          pass_zero=False)

    return b,a

b, a = get_filter(ftype='FIR',
                      order=int(0.3 * float(500)),
                      frequency=[3, 45],
                      sampling_rate=float(500),
                      band='bandpass')

def filter_ecg(ecg=None, sampling_rate=500):
    ecg = np.array(ecg)
    return signal.filtfilt(b, a, ecg)

def resample_ecg(trace, input_freq, output_freq):
    trace = np.atleast_1d(trace).astype(float)
    if input_freq != int(input_freq):
        raise ValueError("input_freq must be an integer")
    if output_freq != int(output_freq):
        raise ValueError("output_freq must be an integer")

    if input_freq == output_freq:
        new_trace = trace
    elif np.mod(input_freq, output_freq) == 0:
        new_trace = decimate(trace, q=input_freq//output_freq,
                             ftype='iir', zero_phase=True, axis=-1)
    else:
        new_trace = resample_poly(trace, up=output_freq, down=input_freq, axis=-1)
    return new_trace
def pad_sequence(x,length):
  pad_x = np.zeros(length,dtype=float)
  for i in range(min(length,len(x))):
    pad_x[i]=x[i]
  return pad_x

# Preprocessing method
def get_transformed_data(ecg_raw,fs):
    warnings.filterwarnings("ignore")
    ecg_filtered = []
    for i in range(len(ecg_raw)):
      x = filter_ecg(ecg_raw[i],500)
      ecg_filtered.append(x)
    ecg_new = []
    for i in range(len(ecg_filtered)):
        ecg_new.append(resample_ecg(ecg_filtered[i],int(fs),500))
    ecg_filtered_padded = []
    for i in range(len(ecg_new)):
        ecg_filtered_padded.append(pad_sequence(ecg_new[i],5000))
    ecg_final = []
    for i in range(len(ecg_filtered_padded)):
      x = (ecg_filtered_padded[i] - np.min(ecg_filtered_padded[i])) / (np.max(ecg_filtered_padded[i]) - np.min(ecg_filtered_padded[i]))
      ecg_final.append(x)
    return np.array(ecg_final)


# Dataset Class

class ECG_Dataset(Dataset):
    def __init__(self, idx_list):
        self.idx_list = idx_list

    def __getitem__(self, index):
        idx = self.idx_list[index]
        file_name = str(idx)+".npy"
        data = np.load(os.path.join("data_processed", file_name))
        label = np.load(os.path.join("labels_processed", file_name))
        x = torch.from_numpy(data).float()
        y = torch.from_numpy(label).float()
        return x , y
    def __len__(self):
        return len(self.idx_list)