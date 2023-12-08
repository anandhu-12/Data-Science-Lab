import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
df = pd.read_csv('C:\\Users\\admin\\Desktop\\Social_Network_Ads.csv')
X = df.iloc[:, [2,3]]
Y = df.iloc[:, 4]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#using linear kernel
from sklearn.svm import SVC
lin = SVC(kernel = 'linear', random_state = 0)
lin.fit(X_train, Y_train)
pred = lin.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,pred))

#using rbf kernel
from sklearn.svm import SVC
rbf = SVC(kernel = 'rbf', random_state = 0)
rbf.fit(X_train, Y_train)
pred = rbf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,pred))

#using poly kernel
from sklearn.svm import SVC
poly = SVC(kernel = 'poly', degree=4)
poly.fit(X_train, Y_train)
pred = poly.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,pred))

plt.scatter(X_test[:, 0], X_test[:, 1],c=Y_test)  

# Create the hyperplane
w = lin.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (lin.intercept_[0]) / w[1]

# Plot the hyperplane
plt.plot(xx, yy)
plt.axis("off"), plt.show();
