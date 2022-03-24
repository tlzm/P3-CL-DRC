import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.fft import fft, fftfreq

bearings = ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_5','Bearing1_6','Bearing1_7']
specialbearing = ['Bearing1_4']
names = ['hour', 'minute', 'second', 'micro', 'haccel', 'vaccel']

def get_rms(records):

    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

num_points = 2560


for bearing in bearings:
    os.chdir('Data_set/' + bearing)
    i = 0
    haccel_rmss, vaccel_rmss,haccel_vars,vaccel_vars,haccel_skews,vaccel_skews = [[] for _ in range(6)]
    haccel_freq_means, vaccel_freq_means = [], []
    haccel_freq_vars, vaccel_freq_vars = [], []
    
    for fname in glob.glob('acc*.csv'):
        # if i % 10 != 0:
        #     i += 1
        #     continue
        df = pd.read_csv(fname, names=names)
        
        freqs = fftfreq(num_points)
        
        haccel_vars.append(df['haccel'].var())
        vaccel_vars.append(df['vaccel'].var())
        haccel_skews.append(df['haccel'].skew())
        vaccel_skews.append(df['vaccel'].skew())

        haccel_rmss.append(get_rms(df['haccel']))
        vaccel_rmss.append(get_rms(df['vaccel']))
        
        
        haccel_fft=np.abs(fft(df['haccel'],num_points))
        vaccel_fft=np.abs(fft(df['vaccel'],num_points))
        
        haccel_fft_mean=np.mean(haccel_fft)
        vaccel_fft_mean=np.mean(vaccel_fft)
        
        haccel_fft_var=np.var(haccel_fft)
        vaccel_fft_var=np.var(vaccel_fft)
        
        haccel_freq_means.append(haccel_fft.mean())
        vaccel_freq_means.append(vaccel_fft.mean())
        haccel_freq_vars.append(haccel_fft.var())
        vaccel_freq_vars.append(vaccel_fft.var())
        
        i += 1
        
    times = [_ for _ in range(len(haccel_rmss))]

    df = pd.DataFrame({'time': times,
                       'haccel_var': haccel_vars,
                       'vaccel_var': vaccel_vars,
                       'haccel_skew': haccel_skews,
                       'vaccel_skew': vaccel_skews,
                       'haccel_rms': haccel_rmss,
                       'vaccel_rms': vaccel_rmss,
                       'haccel_freq_means': haccel_freq_means,
                       'vaccel_freq_means': vaccel_freq_means,
                       'haccel_freq_vars': haccel_freq_vars,
                       'vaccel_freq_vars': vaccel_freq_vars,})


    os.chdir('../..')  # save csv to project directory
    df.to_csv('input/'+bearing + '_RMS.csv',  encoding='utf-8',index = None)

for bearing in specialbearing:
    os.chdir('Data_set/' + bearing)
    i = 0
    haccel_rmss, vaccel_rmss,haccel_vars,vaccel_vars,haccel_skews,vaccel_skews = [[] for _ in range(6)]
    haccel_freq_means, vaccel_freq_means = [], []
    haccel_freq_vars, vaccel_freq_vars = [], []
    
    for fname in glob.glob('acc*.csv'):
        # if i % 10 != 0:
        #     i += 1
        #     continue
        df = pd.read_csv(fname,sep=';', names=names)

        freqs = fftfreq(num_points)
        
        haccel_vars.append(df['haccel'].var())
        vaccel_vars.append(df['vaccel'].var())
        haccel_skews.append(df['haccel'].skew())
        vaccel_skews.append(df['vaccel'].skew())

        haccel_rmss.append(get_rms(df['haccel']))
        vaccel_rmss.append(get_rms(df['vaccel']))
        
        
        haccel_fft=np.abs(fft(df['haccel'],num_points))
        vaccel_fft=np.abs(fft(df['vaccel'],num_points))
        
        haccel_fft_mean=np.mean(haccel_fft)
        vaccel_fft_mean=np.mean(vaccel_fft)
        
        haccel_fft_var=np.var(haccel_fft)
        vaccel_fft_var=np.var(vaccel_fft)
        
        haccel_freq_means.append(haccel_fft.mean())
        vaccel_freq_means.append(vaccel_fft.mean())
        haccel_freq_vars.append(haccel_fft.var())
        vaccel_freq_vars.append(vaccel_fft.var())
        
        i += 1
        
    times = [_ for _ in range(len(haccel_rmss))]

    df = pd.DataFrame({'time': times,
                       'haccel_var': haccel_vars,
                       'vaccel_var': vaccel_vars,
                       'haccel_skew': haccel_skews,
                       'vaccel_skew': vaccel_skews,
                       'haccel_rms': haccel_rmss,
                       'vaccel_rms': vaccel_rmss,
                       'haccel_freq_means': haccel_freq_means,
                       'vaccel_freq_means': vaccel_freq_means,
                       'haccel_freq_vars': haccel_freq_vars,
                       'vaccel_freq_vars': vaccel_freq_vars,})

    os.chdir('../..')  # save csv to project directory
    df.to_csv('input/'+bearing + '_RMS.csv',  encoding='utf-8',index = None)