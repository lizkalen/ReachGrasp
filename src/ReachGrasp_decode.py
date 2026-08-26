## Script to decode Reach&Grasp dataset

# %%
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd


nChann = 64; # Number of HD-sEMG channels
fs = 2000; # Sampling freuqency [Hz]
unit_factor = 1e-3; # factor used to convert mV in V
noverlap = 500
nfft = 4001

# pass band frequency
f_pass = 20
wp = f_pass / (fs/2) # in radians
# stop band frequency
f_stop = 500
ws = f_stop / (fs/2) # in radians

bandpass_order = 2

window = signal.windows.hamming(fs)

bp_b, bp_a = signal.butter(bandpass_order, [20, 500], btype = 'bandpass', fs = fs) # does output='sos' belong here?


# notch filter at 50Hz
notch = 50
bw = 0.2 # bandwidth
Q = notch / bw # quality factor, higher = narrower notch

b_n, a_n = signal.iirnotch(notch, Q, fs = fs)

# %%
REPO_DIR  = os.path.abspath(os.path.join(os.getcwd(), '/Users/lizkal/Library/CloudStorage/SynologyDrive-Personal/SData_ReachGrasp'))
INPUT_DIR = os.path.join(REPO_DIR, 'data')

subjs = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
data_type = ['emg', 'motion', 'tactile']
devices = ['sessantaquattro','cometa','vicon','cyberglove','tactileglove']
tasks = ['HO','HC','WP','WS','WF','WE','Cyl','Sph','Trid','Thumb','FroRea','ReaCyl','ReaSph','Screw','Pour','EatFruit']

n_subj = len(subj)
n_tasks = len(tasks)

# %%
all_task_df = {}

for task in tasks:
    subj_dfs = []
    for subj in subjs:
        task_file_name = os.path.join(INPUT_DIR, 
                                      subj, 
                                      f'/emg/{subj}_task-{task}_acq-sessantaquattro_emg.csv'
                                      )
        df = pd.read.csv(task_file_name)
        df['subject'] = subj
        subj_dfs.append(df)
    all_task_df[task] = pd.concat(subj_dfs, axis = 0, ignore_index = True)
# %%
