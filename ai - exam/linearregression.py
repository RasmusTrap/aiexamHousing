import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Data preprocessing
data = pd.read_csv('housing.csv')
data = data.dropna()
X = data.drop(columns=['median_house_value'])
y = data['median_house_value']

X = pd.get_dummies(X, columns=['ocean_proximity'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Standardization to the numerical features
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Initiate and fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")


# Un-comment to visualize linear regression
#plt.figure(figsize=(10, 8)) 
#plt.scatter(y_test, y_pred)
#plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--') 
#plt.xlabel("Actual Median House Value")
#plt.ylabel("Predicted Median House Value")
#plt.title("Linear Regression: Actual vs. Predicted")
#plt.show()