import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('housing.csv')

# Data preprocessing
# Drop rows with missing values
data = data.dropna()

# Split the data into features (X) and target (y)
X = data.drop(columns=['median_house_value'])  # Exclude the target variable

y = data['median_house_value']

# Convert categorical variables using one-hot encoding
X = pd.get_dummies(X, columns=['ocean_proximity'], drop_first=True)

# Drop problematic columns representing ocean_proximity
X.drop(columns=['ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'], inplace=True)

# Apply Standardization to the numerical features
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Add a constant term to the predictor variables (required for statsmodels)
X = sm.add_constant(X)

# Create the multiple regression model
model = sm.OLS(y, X)

# Fit the model
results = model.fit()

# Print the summary of the regression analysis
print(results.summary())
