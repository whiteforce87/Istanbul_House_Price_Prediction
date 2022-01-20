import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import math

#Reading from file
data = pd.read_excel('istanbul_house_prices.xlsx')
print(data.head())

#Visualizing prices
dataset = data[['price']]
plt.style.use('seaborn')
dataset.plot(color='green', linewidth=3, figsize=(15,6))
plt.show()

X = data[['floor','_totalroom','has_elevator','dist_to_transportation','sea_view','age',
          'within_site','area','number_of_bathrooms']]
Y = data['price']

#Visualizing data

data.hist(['floor','_totalroom','has_elevator','dist_to_transportation','sea_view','age',
          'within_site','area','number_of_bathrooms','price'], figsize=(18,10))

ContinuousCols=['floor','_totalroom','has_elevator','dist_to_transportation','sea_view','age',
                'within_site','area','number_of_bathrooms']

for predictor in ContinuousCols:
    data.plot.scatter(x=predictor, y='price', figsize=(10,5), title=predictor+" VS "+ 'price')


heatmap = sns.heatmap(data.corr(), annot=True)
plt.show()

# Modelling
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)

#Calculating coefficient values:
coeff_data = pd.DataFrame(regressor.coef_ , X.columns , columns=["Coefficients"])
print(coeff_data)
intercept_data =  pd.DataFrame(regressor.intercept_, X.columns, columns = ["Intercepts"])
print("intercept: ", regressor.intercept_,"\n")
print("Predictions:", "\n",y_pred,"\n")

hist = sns.histplot((Y_test-y_pred), bins=30)
plt.show()

#Checking Accuracy value:
from sklearn.metrics import r2_score
R = r2_score(Y_test , y_pred)
print ("R2-score :",R,"\n")

#Showing errors
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(Y_test, y_pred)
print("Mape  : ", mape)
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - Y_test)))
mse = sklearn.metrics.mean_squared_error(Y_test, y_pred)
print("MSE  : " , mse)
rmse = math.sqrt(mse)
print("RMSE  :" , rmse)


#Plotting thr results
plt.style.use('default')
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(7, 3.5))

plt.plot([0, 1750], [0, 1750], '--k')
plt.scatter(Y_test,y_pred, c='grey', label='Prediction')

plt.xlabel('price')
plt.ylabel('Predicted price')
plt.title('Multilinear Regression')
plt.legend()
plt.show()

print("-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-")

#Calculating the estimated house price by entering feature values on UI
def get_regression_predictions(input_features,intercept,slope):
    predicted_values = input_features*slope + intercept
    return predicted_values

floor=int(input("floor:"))
_totalroom=int(input("total room:"))
has_elevator=int(input("has elevator?(0 or 1): "))
dist_to_transportation=float(input("distance to transportation (as km): "))
sea_view=int(input("sea view (0 or 1): "))
age= int(input("age"))
within_site=int(input("within site? (0 or 1): "))
area=int(input("gross area (as m2): "))
number_of_bathrooms =int(input("number of bathrooms: "))

pred_floor = get_regression_predictions(floor,regressor.intercept_,regressor.coef_[0])
pred_totalroom = get_regression_predictions(_totalroom,regressor.intercept_,regressor.coef_[1])
pred_elevator= get_regression_predictions(has_elevator,regressor.intercept_,regressor.coef_[2])
pred_transportation = get_regression_predictions(dist_to_transportation,regressor.intercept_,regressor.coef_[3])
pred_sea_view = get_regression_predictions(sea_view,regressor.intercept_,regressor.coef_[4])
pred_age = get_regression_predictions(age,regressor.intercept_,regressor.coef_[5])
pred_site = get_regression_predictions(within_site,regressor.intercept_,regressor.coef_[6])
pred_area = get_regression_predictions(area,regressor.intercept_,regressor.coef_[7])
pred_bathrooms = get_regression_predictions(number_of_bathrooms,regressor.intercept_,regressor.coef_[8])
KFE = 1.40 # house price ratio taken from government internet site which belongs to June 2021 to October 2021

Estimated_Price=((pred_floor+pred_totalroom+pred_elevator+pred_transportation+
                  pred_sea_view+pred_age+pred_site+pred_area+pred_bathrooms)/9)*KFE
print("Estimated_Price: ", Estimated_Price)






