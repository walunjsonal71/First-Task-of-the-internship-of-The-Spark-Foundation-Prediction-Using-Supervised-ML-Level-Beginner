# First-Task-of-the-internship-of-The-Spark-Foundation-Prediction-Using-Supervised-ML-Level-Beginner
# Task : Prediction Using Supervised ML (Level Beginner)
# The Sparks Foundation (Data Science & Business Analytics Tasks)


# Importing all libraries required for the task
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline


# Reading data
data = pd.read_csv(r"C:\\Users\\admin\\Documents\\Task One Data of The Sparks Foundation.csv")
data.head(10)


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# Preparing the Data
X = data.iloc[:, :-1].values  
Y = data.iloc[:, 1].values  


# Splitting the data into training and testing datasets
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                            test_size=0.2, random_state=0) 
                            
                            
#Training the Algorithm
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, Y_train) 

print("Training Done.")


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# Making Predictions 
print(X_test) # Testing data (In Hours)
Y_pred = regressor.predict(X_test) # Predicting the scores


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
df 


# predicted score if a student studies for 9.25 hrs/ day
hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

# Evaluating the Model
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(Y_test, Y_pred)) 
          
