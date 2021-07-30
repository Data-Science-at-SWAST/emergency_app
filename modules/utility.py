'''
swast_forecast.utility

utilities to help with forecasting

'''

import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from modules.ensemble import ProphetARIMAEnsemble


def pre_process_daily_data(path, observation_col, index_col):
    '''
    Assumes daily data is stored in long format.  Read in 
    and pivot to wide format so that there is a single 
    colmumn for each regions time series.

    Parameters:
    --------
    path: str
        directory and file name of raw datafile

    observation_col: str
        the name of the column containing the time series observations

    index_col: str
         - the index columns.  Assume this is a date.

    Returns:   
    --------
        pd.DataFrame
    '''
    df = pd.read_csv(path, index_col=index_col, parse_dates=True)
    df.columns = map(str.lower, df.columns)
    df.index.rename(str(df.index.name).lower(), inplace=True)
    
    clean_table = pd.pivot_table(df, values=observation_col.lower(), 
                                 index=[index_col.lower()],
                                 columns=['ora'], aggfunc=np.sum)
    
    clean_table.index.freq = 'D'
    
    return clean_table




def get_best_arima_parameters():
    '''
    Returns a dict containing the best arima
    parmaeters for the SWAST forecasting.
    
    A new ambulance trust should run auto_arima
    on their data to get their best parameters and 
    also CV a few models.
    
    Returns:
    -------
        dict
    
    '''
    params = {'order':(1,1,3),
              'seasonal_order':(1,0,1,7)  
    }
    return params


def default_ensemble():
    '''
    Convenience function to create a ProphetARIMAEnsemble
    using default best known parameters.  
    '''
    params = get_best_arima_parameters()
    return ProphetARIMAEnsemble(order=params['order'], 
                                seasonal_order=params['seasonal_order'])


def forecast(y_train, horizon, alpha=0.05, return_all_models=False):
    '''
    Convenience function. All in one forecast function.  
    Create a default ensemble fit the training data and predict ahead.
    
    Parameters:
    --------
    y_train: pd.Series or pd.DataFrame
        y observations for training. Index is a DateTimeIndex
        
    horizon: int
        forecast horizon
        
    alpha: float, optional (default=0.05)
        100(1-alpha) prediction interval
        
    return_all_models: bool. optional (default=False)
        Return individual Regression with ARIMA error and Prophet 
        model predictions 
        
    Returns:
    -------
        pd.DataFrame
    '''
    model = default_ensemble()
    model.fit(y_train)
    return model.predict(horizon, alpha=alpha, 
                         return_all_models=return_all_models)


def multi_region_forecast(y_train, horizon, alpha=0.05, 
                          return_all_models=False):
    '''
    Run forecasts for all regions included in the training data.
    
    This function exploits multiple CPUs. E.g. If your machine has 4 Cores then
    It will run 4 regional forecasts in parrallel.  
    
    Parameters:
    --------
    y_train: pd.DataFrame
        Training data - each colunm in a region.
        
    horizon: int
        forecast horizon
        
    alpha: float, optional (default=0.05)
        100(1-alpha) prediction interval
        
    return_all_models: bool. optional (default=False)
        Return individual Regression with ARIMA error and Prophet 
        model predictions 
        
    Returns:
    -------
        List 
        Each item is a pd.DataFrame containing the forecast for a region.
        Items are ordered by columns in y_train

    '''
    regions = y_train.columns.to_list()
    results = Parallel(n_jobs=-1)(delayed(forecast)(y_train[region], horizon) 
                                                    for region in regions)
    return results
