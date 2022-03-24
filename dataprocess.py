from docutils.parsers.rst.directives import positive_int
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler

bearings = ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7']
cut = [1317,826,1334,1084,2411,1631,2206]

for bearing in bearings:
    dfin = pd.read_csv('input/' + bearing + '_RMS.csv')
    pos = bearings.index(bearing)

#    df = pd.DataFrame({'time': dfin['time']-cut[pos],
    df = pd.DataFrame({'time': dfin['time'],
                       'haccel_var': dfin['haccel_var'],
                       'vaccel_var': dfin['vaccel_var'],
                       'haccel_skew': dfin['haccel_skew'],
                       'vaccel_skew': dfin['vaccel_skew'],
                       'haccel_rms': dfin['haccel_rms'],
                       'vaccel_rms': dfin['vaccel_rms'],
                       'haccel_freq_vars': dfin['haccel_freq_vars'],
                       'vaccel_freq_vars': dfin['vaccel_freq_vars'],
                       'haccel_freq_means': dfin['haccel_freq_means'],
                       'vaccel_freq_means': dfin['vaccel_freq_means'],
                       'RUL': (len(dfin)-1) - dfin['time']})
#    df['time'] = np.where(df['time'] < 0, -1, df['time'] )
    df['RUL'] = np.where(df['RUL'] >= 829, -1, df['RUL'] )
    # MinMax normalization (from 0 to 1)
    cols_normalize = df.columns.difference(['RUL','time'])
    scaler = MinMaxScaler()
    # scaler = MaxAbsScaler()
    # scaler = StandardScaler()
    norm_data = pd.DataFrame(scaler.fit_transform(df[cols_normalize]),columns=cols_normalize, index=df.index)
    join_data = df[df.columns.difference(cols_normalize)].join(norm_data)
    df = join_data.reindex(columns = df.columns)
    
    df=df[~(df['time'].isin([-1]))]
    df=df[~(df['RUL'].isin([-1]))]

    df.to_csv('input/'+bearing + '_set.csv',  encoding='utf-8',index = None)





