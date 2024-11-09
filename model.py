import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math
import xgboost as xgb

base =  r'C:\Users\HP\Desktop\Real word project\turbine\Turbine-renu\model2.pkl'
reg = xgb.XGBRegressor()
# model = pickle.load(open(f'{base}/model2.pkl', "rb"))
model = pickle.load(open(base, "rb"))


def create_f(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    
    return df


def create_pd(fr,to):
    new = pd.date_range(fr+' 12:00:00+00:00',to+' 12:00:00+00:00', freq='10min')
    new = pd.DataFrame(index=new)
    new = create_f(new)
    return new