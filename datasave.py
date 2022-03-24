from numpy.core.numeric import ones
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler

bearings = ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_5','Bearing1_6','Bearing1_7']
specialbearing = ['Bearing1_4']
allbearings = ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7']
names = ['hour', 'minute', 'second', 'micro', 'haccel', 'vaccel']

for bearing in bearings:
    os.chdir('Data_set/' + bearing)
    i = 0
    haccel, vaccel,times = [[] for _ in range(3)]
    bearing_data=np.array([])
    for fname in glob.glob('acc*.csv'):
        # if i % 10 != 0:
        #     i += 1
        #     continue
        df = pd.read_csv(fname, names=names)
        data = np.array(df.loc[:])
#        print(data.shape)
        data=data[:,4:6]
        i += 1
        time = i * ones(1)
        data=np.insert(data, 2, time, axis=1)
#        print(data.shape)
        

        if bearing_data.size == 0:
            bearing_data = data
        else:
            bearing_data = np.append(bearing_data,data,axis=0)


    #df = pd.DataFrame({'time': times,})

    bearing_csv = pd.DataFrame(bearing_data)
    os.chdir('../..')  # save csv to project directory
    bearing_csv.to_csv('input/'+bearing + '_rawdata.csv',  encoding='utf-8',index = None)

for bearing in specialbearing:
    os.chdir('Data_set/' + bearing)
    i = 0
    haccel, vaccel,times = [[] for _ in range(3)]
    bearing_data=np.array([])
    for fname in glob.glob('acc*.csv'):
        # if i % 10 != 0:
        #     i += 1
        #     continue
        df = pd.read_csv(fname,sep=';', names=names)
        data = np.array(df.loc[:])
#        print(data.shape)
        data=data[:,4:6]
        i += 1
        time = i * ones(1)
        data=np.insert(data, 2, time, axis=1)
#        print(data.shape)
        

        if bearing_data.size == 0:
            bearing_data = data
        else:
            bearing_data = np.append(bearing_data,data,axis=0)


    #df = pd.DataFrame({'time': times,})

    bearing_csv = pd.DataFrame(bearing_data)
    os.chdir('../..')  # save csv to project directory
    bearing_csv.to_csv('input/'+bearing + '_rawdata.csv',  encoding='utf-8',index = None)

for bearing in allbearings:
    df = pd.read_csv('input/'+bearing + '_rawdata.csv')

    n_time = df['2'].unique().max()    

    df = pd.DataFrame({'id': df['2'],
                       'haccel':df['0'],
                       'vaccel':df['1'],
                       'RUL':  n_time - df['2']})
    
    # Set RUL label
    
#    MinMax normalization (from 0 to 1)
    cols_normalize = df.columns.difference(['RUL','id'])
#    scaler = MinMaxScaler()
    scaler = MaxAbsScaler()
#    scaler = StandardScaler()
    norm_data = pd.DataFrame(scaler.fit_transform(df[cols_normalize]),columns=cols_normalize, index=df.index)
    join_data = df[df.columns.difference(cols_normalize)].join(norm_data)
    df = join_data.reindex(columns = df.columns)

    df['RUL'] = np.where(df['RUL'] >= 801, 801, df['RUL'] )
    
    df=df[~(df['RUL'].isin([801]))]

    df.to_csv('input/'+bearing + '_rawdata_withlabel.csv',  encoding='utf-8',index = None)


d11 = pd.read_csv('input/Bearing1_1_rawdata_withlabel.csv')
d12 = pd.read_csv('input/Bearing1_2_rawdata_withlabel.csv')
d13 = pd.read_csv('input/Bearing1_3_rawdata_withlabel.csv')
d14 = pd.read_csv('input/Bearing1_4_rawdata_withlabel.csv')
d15 = pd.read_csv('input/Bearing1_5_rawdata_withlabel.csv')
d16 = pd.read_csv('input/Bearing1_6_rawdata_withlabel.csv')
d17 = pd.read_csv('input/Bearing1_7_rawdata_withlabel.csv')


d11 = pd.DataFrame({'id': d11['id']-2002,'haccel':d11['haccel'],'vaccel':d11['vaccel'],'RUL':d11['RUL']})
d12 = pd.DataFrame({'id': d12['id']-70,'haccel':d12['haccel'],'vaccel':d12['vaccel'],'RUL':d12['RUL']})
d13 = pd.DataFrame({'id': d13['id']-1574,'haccel':d13['haccel'],'vaccel':d13['vaccel'],'RUL':d13['RUL']})
d14 = pd.DataFrame({'id': d14['id']-627,'haccel':d14['haccel'],'vaccel':d14['vaccel'],'RUL':d14['RUL']})
d15 = pd.DataFrame({'id': d15['id']-1662,'haccel':d15['haccel'],'vaccel':d15['vaccel'],'RUL':d15['RUL']})
d16 = pd.DataFrame({'id': d16['id']-1647,'haccel':d16['haccel'],'vaccel':d16['vaccel'],'RUL':d16['RUL']})
d17 = pd.DataFrame({'id': d17['id']-1458,'haccel':d17['haccel'],'vaccel':d17['vaccel'],'RUL':d17['RUL']})

d11.to_csv('input/LearningSet1.csv',  encoding='utf-8',index = None)
d12.to_csv('input/LearningSet2.csv',  encoding='utf-8',index = None)
d13.to_csv('input/LearningSet3.csv',  encoding='utf-8',index = None)
d14.to_csv('input/LearningSet4.csv',  encoding='utf-8',index = None)
d15.to_csv('input/LearningSet5.csv',  encoding='utf-8',index = None)
d16.to_csv('input/LearningSet6.csv',  encoding='utf-8',index = None)
d17.to_csv('input/LearningSet7.csv',  encoding='utf-8',index = None)

""" k=0
for bearing in allbearings:
    df = pd.read_csv('input/'+bearing + '_rawdata.csv')
    n_time = df['2'].unique().max()    
    df = pd.DataFrame({'id': df['2'],
                       'haccel':df['0'],
                       'vaccel':df['1'],
                       'RUL':  n_time - df['2']})
    
#    MinMax normalization (from 0 to 1)
    cols_normalize = df.columns.difference(['RUL','id'])
#    scaler = MinMaxScaler()
    #scaler = MaxAbsScaler()
    scaler = StandardScaler()
    norm_data = pd.DataFrame(scaler.fit_transform(df[cols_normalize]),columns=cols_normalize, index=df.index)
    join_data = df[df.columns.difference(cols_normalize)].join(norm_data)
    df = join_data.reindex(columns = df.columns)
    # Set RUL label
    df['RUL'] = np.where(df['RUL'] >= 800, 800, df['RUL'] )
    k=k+1
    df.to_csv('input/LearningSet'+str(k)+'.csv',  encoding='utf-8',index = None) """