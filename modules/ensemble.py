import io
import sys
import numpy as np
import pandas as pd
from fbprophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast_tools.metrics import mean_absolute_error
from forecast_tools.model_selection import (rolling_forecast_origin, forecast_accuracy, cross_validation_score)
from forecast_tools.baseline import SNaive



class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    Credit to SkLearn NotFittedError"""


def new_years_day_holidays(start='2013-01-01', years=20):
    '''
    A convenience function for creating a Prophet format 
    new years day holidays dataframe.

    Bu default returns new years day dates between 2013 and
    20 years into the future.

    Parameters:
    ---------
    start: str, optional (default='2013-01-01')

    years: int, optional (default=20)
        How many years of new years day dates to return from the
        start date

    Returns:
    --------
        pd.DataFrame
        In Prophet format columns = [holiday, ds]

    '''

    return pd.DataFrame({
        'holiday': 'new_year',
        'ds': pd.date_range(start=start,
                            periods=years,
                            freq='YS')
    })


class ProphetARIMAEnsemble():
    '''
    An ensemble of Prophet and Regression with ARIMA errors for forecasting
    the no. of calls that require the dispatch of > 1 ambulance.  

    Works by taking the average of the two forecasting methods.

    Both methods allow the addition of holidays.  For SWAST
    the most important holiday/special event was found to be new years day.

    '''

    def __init__(self, order, seasonal_order, prophet_default_alpha=0.05,
                 **arima_kwargs):
        '''
        Initialise the ensemble

        Parameters:
        ---------
        order: tuple
            ARIMA(p, d, q) p = AR; d = integrated; q = MA. e.g. (3, 1, 0)

        seasonal_order: tuple 
            SARIMA(P, D, Q, s) seasonal equivalent of ARIMA order. s is for the 
            seasonal period.  e.g. 7 for daily data.

        prophet_default_alpha: float, optional (default=0.05)
            Prophet requires an interval_width parameter (1-alpha) on 
            initialisation

        **arima_kwargs: dict
            Dictionary of arima kw arguments

        '''
        self.order = order
        self.seasonal_order = seasonal_order

        # by default the ensemble has new years day as a holiday
        self.holidays = new_years_day_holidays()

        # needed because Prophet constructor
        self.alpha = prophet_default_alpha

        #keyword arguments for fitting of the arima model.
        self.arima_kwargs = arima_kwargs

        # variable representing models in ensemble
        self.arima_model = None
        self.prophet_model = None

        # model not yet fitted
        self._fitted = False

    def __str__(self):
        return f'ProphetARIMAEnsemble(order={self.order}, ' \
            + f'seasonal_order={self.seasonal_order}, ' \
                + 'prophet_default_alpha={self.alpha})'

    def __repr__(self):
        return f'ProphetARIMAEnsemble(order={self.order}, ' \
            + f'seasonal_order={self.seasonal_order}, ' \
                + 'prophet_default_alpha={self.alpha})'

    def fit(self, y_train):
        '''
        Fit the model to the training data

        Parameters:
        ----------
        y_train: pd.DataFrame
            pandas dataframe containing the y training data.
            assumes that the index is a datetimeindex and there
            is a single column of training data 

        '''

        # fit the arima model
        self._fit_arima(y_train)

        # fit prophet
        self._fit_prophet(y_train, self.alpha)

        # fitting complete
        self._fitted = True

    def _fit_arima(self, y_train):
        '''
        Fits an ARIMA model to the training data

        Parameters:
        ----------
        y_train: pd.DataFrame
            time series data.  DataFrame uses a DateTimeIndex
        '''

        # store training index and data
        self._training_index = y_train.index
        self._y_train = y_train

        # encode training holidays for ARIMA
        arima_holidays = self._encode_holidays(self.holidays['ds'].to_numpy(),
                                               y_train.index)

        # create and fit the ARIMA model
        self.arima_model = ARIMA(endog=y_train,
                                 exog=arima_holidays,
                                 order=self.order,
                                 seasonal_order=self.seasonal_order,
                                 enforce_stationarity=False)


        #temp
        self.arima_model = SARIMAX(endog=y_train,
                                   exog=arima_holidays,
                                   order=self.order,
                                   seasonal_order=self.seasonal_order,
                                   enforce_stationarity=False, 
                                   enforce_invertibility=False)

        # fit the ARIMA model...

        # redirect stdout
        arima_fit_log = io.StringIO()
        #sys.stdout = arima_fit_log

        #keyword arguments
        if 'method' in self.arima_kwargs:
            method =  self.arima_kwargs['method']
        else:
            method = 'lbfgs'
        
        if 'maxiter' in self.arima_kwargs:
            maxiter = self.arima_kwargs['maxiter']
        else:
            maxiter = 500

        # fit the ARIMA model
        
        if self._fitted:
            #using existing parameter estimates as starting point for opt
            params = self._arima_fitted.params
            self._arima_fitted = self.arima_model.fit(method=method, 
                                                      maxiter=maxiter,
                                                      start_params=params)
        else:
            self._arima_fitted = self.arima_model.fit(method=method, 
                                                      maxiter=maxiter)

        # now restore stdout function
        #sys.stdout = sys.__stdout__

        #self.arima_fit_log = arima_fit_log
        

    def _fit_prophet(self, y_train, alpha):
        '''
        Fits an Prophet model to the training data

        Parameters:
        ----------
        y_train: pd.DataFrame
            time series data.  DataFrame uses a DateTimeIndex

        alpha: float
            used to form a 100(1 - alpha) prediction interval
        '''
        # minimal options set.
        self.prophet_model = Prophet(holidays=self.holidays,
                                     interval_width=1 - alpha,
                                     daily_seasonality=False)

        self.prophet_model.fit(self._prophet_training_data(y_train))

    def _encode_holidays(self, holidays, idx):
        '''
        Encodes holidays as a SINGLE dummy variable
        for the ARIMA model.

        I.e. 0 if no holiday 1 if holiday important.
        This would need adapting if ARIMA wants to model
        the effect of individual holidays/special days 
        differently.

        Parameters:
        ---------
        holidays: array-like
            list of holidays

        idx: DataTimeIndex
            date times to check

        Returns:
        --------
            pd.DataFrame
            0/1 encoding with a DateTimeIndex.
        '''
        dummy = idx.isin(holidays).astype(int)
        dummy = pd.DataFrame(dummy)
        dummy.columns = ['holiday']
        dummy.index = idx
        return dummy

    def _prophet_training_data(self, y_train):
        '''
        Converts a standard pandas datetimeindexed dataframe
        for time series into one suitable for Prophet

        Parameters:
        ---------
        y_train: pd.DataFrame
            univariate time series data

        Returns:
        --------
            pd.DataFrame in Prophet format 
            columns = ['ds', 'y']

        '''
        prophet_train = pd.DataFrame(y_train.index)
        prophet_train['y'] = y_train.to_numpy()
        prophet_train.columns = ['ds', 'y']

        return prophet_train

    def predict(self, horizon, alpha=0.05, return_pred_int=False,
                return_all_models=False):
        '''
        Produce a point forecast and prediction intervals for a period ahead.


        Paramerters:
        --------
        horizon: int
            periods to ahead to forecast

        alpha: float, optional (default=0.05)
            width of prediction intervals

        return_all_models: bool, optional (default=False)
            If true returns ensemble results AND ARIMA and Prophet 
            point forecasts and prediction intervals

        Returns:
        ---------
            pd.DataFrame

            Forecasted time series in a data frame with a pd.DataTimeIndex.  
            Minimum columns are yhat = mean forecast; 
            yhat_lower_{(1-alpha)*100} = lower PI;
            yhat_upper_{(1-alpha)*100} = upper prediction interval.  

            If return_all_models=True then returns an extended DataFrame 
            with columns for ARIMA and Prophet models.

        '''

        self.check_is_fitted()

        # reg with ARIMA errors prediction
        arima_forecast = self._arima_predict(horizon, alpha)

        # prophet prediction
        prophet_forecast = self._prophet_predict(horizon, alpha)

        # average forecast
        post_fix = str(int((1 - alpha) * 100))

        summary_frame = pd.concat([arima_forecast, prophet_forecast], axis=1)
        summary_frame['yhat'] = summary_frame[[
            'prophet_mean', 'arima_mean']].mean(axis=1)
        summary_frame['yhat_lower_' + post_fix] = \
            summary_frame[['prophet_lower_' + post_fix,
                           'arima_lower_' + post_fix]].mean(axis=1)
        summary_frame['yhat_upper' + post_fix] = \
            summary_frame[['prophet_upper_' + post_fix,
                           'arima_upper_' + post_fix]].mean(axis=1)

        # sort columns with ensemble predictions first
        sorted_columns = summary_frame.columns[-3:].to_list()
        sorted_columns += summary_frame.columns[:-3].to_list()

        if return_all_models:
            return summary_frame[sorted_columns]
        elif return_pred_int:
            return summary_frame[sorted_columns[:3]]
        else:
            return summary_frame[sorted_columns[:1]]
            

    def _arima_predict(self, horizon, alpha=0.05):
        '''
        ARIMA forecast

        Parameters:
        ---------
        horizon: int
            periods to ahead to forecast

        alpha: float, optional (default=0.05)
            width of prediction intervals

        Returns:
        ---------
        pd.DataFrame

        Columns = Mean, Lower_PI, Upper_PI

        '''

        # equivalent to prophet make future dataframe
        pred_idx = pd.date_range(start=self._training_index[-1],
                                 periods=horizon+1,
                                 freq=self._training_index.freq)[1:]

        # encode holidays for prediction
        exog_holiday = self._encode_holidays(self.holidays['ds'].to_numpy(),
                                             pred_idx)

        forecast = self._arima_fitted.get_forecast(horizon, exog=exog_holiday)

        df = forecast.summary_frame(alpha=alpha)
        df = df[['mean', 'mean_ci_lower', 'mean_ci_upper']]
        post_fix = str(int((1 - alpha) * 100))
        df.columns = ['arima_mean', 'arima_lower_' +
                      post_fix, 'arima_upper_' + post_fix]
        df.index.name = 'ds'
        return df

    def _prophet_predict(self, horizon, alpha):
        '''

        Prophet forecast

        Parameters:
        ---------
        horizon: int
            periods to ahead to forecast

        alpha: float
            width of prediction intervals

        Returns:
        ---------
        pd.DataFrame

        Columns = Mean, Lower_PI, Upper_PI

        '''
        # Prophet needs to be refitted if alpha is different from default
        if alpha != self.alpha:
            self._fit_prophet(self._y_train, alpha)

        future = self.prophet_model.make_future_dataframe(periods=horizon)
        prophet_forecast = self.prophet_model.predict(future)
        df = prophet_forecast[['ds', 'yhat',
                               'yhat_lower', 'yhat_upper']][-horizon:]
        post_fix = str(int((1 - alpha) * 100))
        df.columns = ['ds', 'prophet_mean', 'prophet_lower_' +
                      post_fix, 'prophet_upper_' + post_fix]
        df.index = df['ds']
        return df.drop(['ds'], axis=1)

    def check_is_fitted(self):
        if not self._fitted:
            msg = 'Ensemble model has not been fitted. ' \
                + 'Please call model.fit(y_train)'
            raise NotFittedError(msg)

            
def ensemble_cross_val_score(model, data, horizons, metric,
                             min_train_size='auto', step=7):
    '''
    Cross validation of the ensemble.
    
    Please note this function does not contain any validation
    of inputs at this present time.
    
    Parameters:
    -----------
    data:  pd.Series or np.array
        a univariate time series
        
    horizons: list
        e.g. [7, 14, 28] a int list of sub-horizons to cross-validate.
        Increasing the maximum forecast horizon results in less cv folds.
       
    metric: object
        function with signature (y_true, y_pred) used to calculate a
        forecast error
        
    min_train_size: str or int. optional (default = 'auto')
        By default the function uses the last 84 days of the data for CV
        E.g. data.shape[0] - 365 (auto setting.).  Alternatively use
        an interger value to manually set.  Increasing the min_train_size
        results in less folds
    
    step: int
        Spacing between splits. Increasing the spacing leads to less folds.
    
    
    Returns:
    --------
        pd.DataFrame
        Columns = horizons, 
        Rows = Splits
        Value = forecast error calculated using @metric
    
    '''

    if min_train_size == 'auto':
        min_train_size = data.shape[0] - 365
    
    #maximum horizon
    max_horizon = max(horizons)
    
    #predict up to max_horizomn days ahead spacing each split by step
    cv = rolling_forecast_origin(data, 
                                 min_train_size=min_train_size, 
                                 horizon=max_horizon, 
                                 step=step)

    #get MAE results for cv results for sub-horizons using all virtual cores 
    results_1 = cross_validation_score(model, cv, 
                                       horizons=horizons,
                                       metric=mean_absolute_error, 
                                       n_jobs=1)
    
    #return results as a dataframe
    return pd.DataFrame(results_1, columns=horizons)
