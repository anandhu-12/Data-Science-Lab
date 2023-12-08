import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
print(diabetes.feature_names)

# Use only one feature (column) for simplicity
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = reg.predict(diabetes_X_test)

# Print the coefficients and intercept
print("Coefficients: \n", reg.coef_)
print("Intercept: \n", reg.intercept_)

# Print mean squared error
print("Mean Squared Error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# Print coefficient of determination (r2 score)
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xlabel("Age")
plt.ylabel("Diabetes Progression")
plt.xticks(())
plt.yticks(())

plt.show()
