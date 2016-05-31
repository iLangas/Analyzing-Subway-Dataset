import numpy as np
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv('C:\\Users\\Elias\\Desktop\\Data_analyst_nanodegree\\P1_Analyzing_the_NYC_subway_dataset\\Final_project\\turnstile_data_master_with_weather-utf8.csv')

# the histogram of the data
#plt.hist() cannot plot both histograms--error occurs with hash reference
#hist() method though when cast onto DataFrame object doesn't cause conflict
#n, bins, patches = plt.hist(x, 50, facecolor='g', alpha=0.75, label = 'without rain')
#n, bins, patches = plt.hist(y, 50, facecolor='r', alpha=0.75, label = 'with rain')

plt.figure();
x = df['ENTRIESn_hourly'][df['rain'] == 0]
y = df['ENTRIESn_hourly'][df['rain'] == 1]
#total number of riders with and without rain
summ_rain = np.sum(df['ENTRIESn_hourly'][df['rain'] == 1], axis = 1)
summ_norain = np.sum(df['ENTRIESn_hourly'][df['rain'] == 0], axis = 1)
print(summ_norain, summ_rain)

#Creation of histogram
x.hist(alpha = 0.5, bins = 50, label = 'No rain')
y.hist(alpha = 0.8, bins = 50, label = 'Rain')

#Axis and graph adaptation
plt.xlabel('ENTRIESn_houlry')
plt.ylabel('Number of records')
plt.title('Entries in regard to rainfall')
plt.axis([0, 7000, 0, 70000])
plt.legend(loc='upper right')
#plt.grid(True)
plt.show()