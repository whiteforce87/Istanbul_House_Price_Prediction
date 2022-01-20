from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_excel('istanbul_house_prices.xlsx')
X = dataset[['floor','_totalroom','has_elevator','dist_to_transportation','sea_view','age',
          'within_site','area','number_of_bathrooms']]
y = dataset['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

rgr = MLPRegressor(solver='adam',activation='relu',verbose=1 ,hidden_layer_sizes=(32,32,32), max_iter=350,
                       random_state=2,alpha=0.0001, batch_size=10,learning_rate_init=0.001,power_t=0.5)


rgr.fit(X_train, y_train)
y_pred = rgr.predict(X_test)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from math import sqrt

R = r2_score(y_test , y_pred)
print ("R2-score :",R)
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - y_test)))
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mape  : ", mape)
train_mse = mean_squared_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
train_preds = rgr.predict(X_test)
mse = mean_squared_error(y_test, train_preds)
rmse = sqrt(mse)
print("mse is: ", mse)
print("rmse is: " , rmse)

print("best loss: ", rgr.best_loss_)
print("Train MSE:", np.round(train_mse,2))
print("Test MSE:", np.round(test_mse,2))

plt.plot([0, 1750], [0, 1750], '--k')
plt.scatter(y_test,y_pred, c='pink', label='Prediction')
plt.xlabel('price')
plt.ylabel('Predicted price')
plt.title('MLP Regression')
plt.legend()
plt.show()

plt.plot(rgr.loss_curve_, label= 'Train')
rgr.fit(X_test,y_test)
plt.plot(rgr.loss_curve_, label = 'Test')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()
