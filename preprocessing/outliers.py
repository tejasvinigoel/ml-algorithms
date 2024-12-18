#If a Z-score is 0.0, it indicates that the data point's score is identical to the mean score.
# A Z-score of 1.0 represents a value that is one standard deviation from the mean. 
# Z-scores may be positive or negative, with a positive value indicating the score is above the mean 
# negative score indicating it is below the mean.

import numpy as np
data = titanic_df['fare']
mean = np.mean(data) # calculates the mean
std_dev = np.std(data) # calculates the standard deviation
Z_scores = (data - mean) / std_dev # computes the Z-scores
outliers = data[np.abs(Z_scores) > 3] # finds all the data points that are 3 standard deviations away from the mean

#iqr-- the range within which the central half of the data lies
Q1 = titanic_df['fare'].quantile(0.25) # calculates the first quartile
Q3 = titanic_df['fare'].quantile(0.75) # calculates the third quartile
IQR = Q3 - Q1 # computes the IQR

# Below, we find all the data points that fall below the lower bound or above the upper bound
outliers = titanic_df['fare'][
    (titanic_df['fare'] < (Q1 - 1.5 * IQR)) |
    (titanic_df['fare'] > (Q3 + 1.5 * IQR))
]

#method 3---SD
mean = np.mean(titanic_df['fare']) # calculates the mean
standard_deviation = np.std(titanic_df['fare']) # calculates the standard deviation
outliers = titanic_df['fare'][np.abs(titanic_df['fare'] - mean) > 3 * standard_deviation] # finds all the data points that are 3 standard deviations away from the mean
