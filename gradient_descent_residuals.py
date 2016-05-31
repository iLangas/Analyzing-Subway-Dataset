import numpy as np
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv('C:\\Users\\Elias\\Desktop\\Analyzing_subway_dataset\\data_with_weather.csv')

def normalize_features(array):
   """
   Normalize the features in the data set.
   """
   array_normalized = (array-array.mean())/array.std()
   mu = array.mean()
   sigma = array.std()

   return array_normalized, mu, sigma

def compute_cost(features, values, theta):
    
    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)

    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    
    m = len(values)
    convergence = alpha/len(values)
    cost_history = [None]*num_iterations
    y =[None]*len(theta)

    for i in range(num_iterations):
        y = np.dot((np.dot(features, theta) - values),features)
        theta = theta - convergence*y
        cost_history[i] = compute_cost(features, values, theta) 
    return theta, pandas.Series(cost_history)

def predictions(dataframe):
    
    # Select Features (try different features!)
    features = dataframe[['rain', 'precipi']]
    
    # Add UNIT and Hour to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    dummy_units = pandas.get_dummies(dataframe['Hour'], prefix='hour')
    features = features.join(dummy_units)
    #dummy_units = pandas.get_dummies(dataframe['DATEn'], prefix='date')
    #features = features.join(dummy_units)
    
    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)

    # Set values for alpha, number of iterations.
    alpha = 0.1 # please feel free to change this value
    num_iterations = 30 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    
    predictions = np.dot(features_array, theta_gradient_descent)
    
    return predictions




    
resid = np.asarray(df['ENTRIESn_hourly']).T - predictions(df)
    
x_axis = [i for i in range(1,len(df['ENTRIESn_hourly'])+1)]
plt.plot(x_axis, resid, 'ro')
plt.xlabel('Case ID')
plt.ylabel('Residual error')
plt.title('Residual graph')
plt.show()
