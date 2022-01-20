# K-Nearest Neighbor(KNN)
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn import neighbors

#Reading from file
data = pd.read_excel('istanbul_House_prices_main.xlsx')
print(data.head())

X = data[['floor','_totalroom','has_elevator','sea_view','age',
          'within_site','area','number_of_bathrooms']]
y = data['price']

TargetVariable='price'
Predictors=['floor','_totalroom','has_elevator','dist_to_transportation','sea_view','age',
          'within_site','area','number_of_bathrooms']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=10)

from sklearn.neighbors import KNeighborsRegressor
RegModel = KNeighborsRegressor(n_neighbors=3)

# Printing all the parameters of KNN
print(RegModel)

# Creating the model on Training Data
KNN=RegModel.fit(X_train,y_train)
prediction=KNN.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, KNN.predict(X_train)))

###########################################################################

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

n_neighbors=3

for i, weights in enumerate(["uniform", "distance"]):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    prediction = knn.fit(X_train,y_train).predict(X_train)

    plt.subplot(2, 1, i + 1)
    plt.scatter(y_train,prediction, color="darkorange", label="prediction")
    plt.plot([0, 1750], [0, 1750], '--k')
    plt.axis("tight")
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.tight_layout()
plt.show()

