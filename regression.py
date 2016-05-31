import numpy as np
import pandas
import scipy
import statsmodels.api as sm
from ggplot import *

weather_turnstile = pandas.read_csv('C:\\Users\\Elias\\Desktop\\Analyzing_subway_dataset\\data_with_weather.csv')

def normalize_features(array):
    """
    Normalize the features in the data set.
    """
    array_normalized = (array-array.mean())/array.std()
    mu = array.mean()
    sigma = array.std()
    
    return array_normalized, mu, sigma

def linear_regression_OLS(Y,X):
    #We dont benefit from normalization in this method. We can ignore it.
    X, mu, sigma = normalize_features(X)
    
    #Add constant column
    X = sm.add_constant(X)
    
    #Apply model and take results
    model = sm.OLS(Y, X)
    results = model.fit()
    #print(results.summary())
    results = results.params
    
    #Prepare data for prediction calculation. Our model should have the form y = b0 + b1*x
    #For example :
    #prediction = parameters[0] + parameters[1]*X['Hour'] + results.params[2]*X['rain'] + ...
    param = np.matrix(results)
    param  = np.tile(param, (len(X), 1))
    trans_param = np.asarray(param).T
    
    #Prediction calculation
    pred = np.sum(np.multiply(param,X), axis = 1)
    
    # Transform matrix into 1-dimensional array
    pred_array = np.squeeze(np.asarray(pred))
    #pred_array = np.asarray(pred).reshape(-1)
    
    #Pass prediction to output
    prediction = pred_array 
    
    return prediction

def polynomial_regression(X, Y, ORDER):
    pol_reg = np.polyfit(X, Y, ORDER)
    polynomial = np.poly1d(pol_reg)
    pol = polynomial(X)
    print(polynomial)
    print(pol_reg)
    print(len(pol))
    param = np.matrix(pol_reg)
    param  = np.tile(param, (len(X), 1))
    print(param)
    trans_param = np.asarray(param).T
    
    #Prediction calculation
    pred = np.sum(np.multiply(param,X), axis = 1)
    
    # Transform matrix into 1-dimensional array
    pred_array = np.squeeze(np.asarray(pred))
    #pred_array = np.asarray(pred).reshape(-1)
    
    #Pass prediction to output
    prediction = pred_array
    
    return prediction

def predictions(weather_turnstile):
    
    #feature selection
    d = [0, 1]
    Regression_coefficients = weather_turnstile[['rain']]
    '''dummy_units = pandas.get_dummies(weather_turnstile['UNIT'], prefix='unit')
    Regression_coefficients = Regression_coefficients.join(dummy_units)
    dummy_units = pandas.get_dummies(weather_turnstile['Hour'], prefix='hour')
    Regression_coefficients = Regression_coefficients.join(dummy_units)
    dummy_units = pandas.get_dummies(weather_turnstile['DATEn'], prefix='date')
    Regression_coefficients = Regression_coefficients.join(dummy_units)'''
    
    #check linearity of regression coefficients to ENTRIEn_hourly
    plot = ggplot(weather_turnstile, aes(x =weather_turnstile['Hour'] , y = weather_turnstile['ENTRIESn_hourly'])) + geom_line()
    #print(plot)
    plt.show()
    
    #OLS
    #prediction = linear_regression_OLS(weather_turnstile['ENTRIESn_hourly'], Regression_coefficients)
    
    #polynomial Regression
    x = np.array(weather_turnstile['Hour'])
    y = np.array(weather_turnstile['ENTRIESn_hourly'])
    order = 0
    
    prediction = polynomial_regression(x, y, order)
    
    #prediction = np.polyfit(x, y, order)
   
        
    return prediction