"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from scipy import stats
#preprocess
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler

import logging
"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)
"""
def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')

def norm_series(series):
    y = series.resample('MS').mean()
    values2 = y.values
    values2 = values2.reshape((len(values2), 1))
    # train the normalization
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = scaler2.fit(values2)
    ser2 = DataFrame(y)
    ser2['normalized']= scaler2.transform(values2)
    ser2['log']=np.log(values2 )
    ser2['loglog']=np.log(np.log(values2 ))
    ser2.head(4)
    return ser2

def get_cifar10(type_ser):
    """Retrieve the CIFAR dataset and process the data."""
  
    #print('Total number of outputs : ', nClasses)
    #print('Output classes : ', classes)
    # Get the data.
    #filename='india_all_stations_label.csv'
    filename='433440-99999-merge.csv'

    #filename='india_all_stations_comma.txt'
    series = read_csv( filename, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    
    series=series.dropna(axis=0)
    #X = series#.iloc[:,:]  # independent variables
    ser2=norm_series(series)
    if type_ser=='normal':
        return(ser2['TEMP'])
    elif type_ser =='normalized':
        return(ser2['normalized'])
    elif type_ser =='log':
        return(ser2['log'])
    elif type_ser =='loglog':
        return(ser2['loglog']) 
    #return np.log(np.log(X))

def compile_model_mlp(network, X):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    p_values  = network['p_values' ]
    d_values  = network['d_values']
    q_values  = network['q_values']
    seasonal_p= network['sp_values']
    seasonal_d= network['sd_values']
    seasonal_q= network['sq_values']
    #optimizer  = geneparam['optimizer' ]

    #logging.info("Architecture:%d,%d,%d" % (p_values, d_values, q_values))
        
    X = X.astype('float32')
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    param = (p_values, d_values, q_values)
    param_seasonal=(seasonal_p, seasonal_d, seasonal_q,12)
    
    logging.info('SARIMAX: {} x {}'.format(param, param_seasonal))
    #for t in range(len(test)):
    warnings.filterwarnings("ignore")
        #model = ARIMA(history, order=arima_order)
    mod = sm.tsa.statespace.SARIMAX(X,order=param, seasonal_order=param_seasonal, enforce_stationarity=False,
                                            enforce_invertibility=False)

    model_fit = mod.fit()
        #minResult.append( results.aic )
        #minParam.append(param)
        #minParam_Seasonal.append(param_seasonal)
    #print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, model_fit.aic))
        #model_fit = model.fit(disp=0)
        #yhat = model_fit.forecast()[0]
        #predictions.append(yhat)
        #history.append(test[t])
    # calculate out of sample error
    #error = mean_squared_error(test, predictions)
    #print(model_fit.aic, model_fit.bic)
    return param, param_seasonal, model_fit.aic

def train_and_score(network, dataset,type_ser):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str) : Dataset to use for training/evaluating

    """
    #logging.info("Getting SARIMAX datasets")
    X = get_cifar10(type_ser)
    #logging.info("Compling SARIMAX model")
    warnings.filterwarnings("ignore")
    try:
        param, param_seasonal, aic = compile_model_mlp(network, X.values)
    #if mse!=None:
        #print('aic error:', mse)
        return param, param_seasonal, aic
    except:
        print('aic error:', -1)
        return -1
