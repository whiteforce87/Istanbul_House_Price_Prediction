# Support Vector Machines(SVM)
from matplotlib import pyplot as plt
from sklearn import svm
import pandas as pd
import numpy as np

#Reading from file
data = pd.read_excel('istanbul_house_prices_main.xlsx')
print(data.head())

X = data[['floor','_totalroom','has_elevator','dist_to_transportation','sea_view','age',
          'within_site','area','number_of_bathrooms']]
Y = data['price']

TargetVariable='price'
Predictors=['floor','_totalroom','has_elevator','dist_to_transportation','sea_view','age',
          'within_site','area','number_of_bathrooms']

RegModel = svm.SVR(C=100, kernel='linear', gamma=0.01)

# Printing all the parameters
print(RegModel)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Creating the model on Training Data
SVM=RegModel.fit(X_train,y_train)
prediction=SVM.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, SVM.predict(X_train)))

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
plt.scatter(y_test,prediction, c='r', label='Prediction')

plt.xlabel('price')
plt.ylabel('Predicted price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()