# Decision Trees
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

#Reading from file
#from DecisionTree import feature_cols

data = pd.read_excel('istanbul_house_prices_main.xlsx')
print(data.head())

X = data[['floor','_totalroom','has_elevator','dist_to_transportation','sea_view','age',
          'within_site','area','number_of_bathrooms']]
Y = data['price']

TargetVariable='price'
Predictors=['floor','_totalroom','has_elevator','dist_to_transportation','sea_view','age',
          'within_site','area','number_of_bathrooms']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

RegModel = DecisionTreeRegressor(max_depth=6,criterion='squared_error')
# Good Range of Max_depth = 2 to 20

# Printing all the parameters of Decision Tree
print(RegModel)

# Creating the model on Training Data
DT=RegModel.fit(X_train,y_train)
prediction=DT.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, DT.predict(X_train)))

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
print(TestingDataResults.head())

# Calculating the error
from sklearn.metrics import mean_squared_error
from math import sqrt
train_preds = RegModel.predict(X_train)
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_test, prediction)
print("Mean absolute error: %.2f" % np.mean(np.absolute(prediction - y_test)))
print("Mape  : ", mape)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
print("mse is: ", mse)
print("rmse is: " , rmse)


plt.plot([0, 1750], [0, 1750], '--k')
plt.scatter(y_test,prediction, c='g', label='Prediction')

plt.xlabel('price')
plt.ylabel('Predicted price')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()