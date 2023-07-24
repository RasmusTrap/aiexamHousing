import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

#Data preprocessing
data = pd.read_csv('housing.csv')
data = data.dropna()
X = data.drop(columns=['median_house_value'])  # Exclude the target variable
y = data['median_house_value']
X = pd.get_dummies(X, columns=['ocean_proximity'], drop_first=True)
X.drop(columns=['ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'], inplace=True)

# Apply Standardization to the numerical features
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
