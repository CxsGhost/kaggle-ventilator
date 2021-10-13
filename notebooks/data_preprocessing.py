import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

train_ori = pd.read_csv('../data/train.csv', index_col=None, header=0)


def add_features(df):
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()

    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()

    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    # df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    # df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    # df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    # df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    # df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    # df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    # df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    # df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    # df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    # df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    # df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    # df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    # df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    # df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)

    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')

    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df.drop(['u_in_lag1'], axis=1)
    df.drop(['u_out_lag1'], axis=1)
    # df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    # df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']

    # df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    # df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']

    # df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    # df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    # df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    # df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    df['cross'] = df['u_in'] * df['u_out']
    df['cross2'] = df['time_step'] * df['u_out']

    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)
    return df


train = add_features(train_ori)
targets = train[['pressure']].to_numpy().reshape(-1, 80)
train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
train = train.values

scale = StandardScaler()
train = scale.fit_transform(train)
train = train.reshape(-1, 80, train.shape[-1])

np.save('../data/x_train.npy', train)
np.save('../data/y_train.npy', targets)

