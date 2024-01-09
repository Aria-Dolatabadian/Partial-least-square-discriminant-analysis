import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read data from CSV
data = pd.read_csv('synthetic_data.csv')

# Split the data into features (X) and target (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Perform PLS-DA
n_components = 2
plsda = PLSRegression(n_components=n_components)
plsda.fit(X_train_std, y_train)

# Transform the data using PLS-DA
X_train_transformed = plsda.transform(X_train_std)
X_test_transformed = plsda.transform(X_test_std)

# Visualize the results
plt.scatter(X_train_transformed[y_train == 0, 0], X_train_transformed[y_train == 0, 1], label='Class 0', marker='o')
plt.scatter(X_train_transformed[y_train == 1, 0], X_train_transformed[y_train == 1, 1], label='Class 1', marker='x')
plt.title('PLS-DA Visualization (Training Set)')
plt.xlabel('PLS-DA Component 1')
plt.ylabel('PLS-DA Component 2')
plt.legend()
plt.show()
