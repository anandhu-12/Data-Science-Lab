import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
mydata = pd.read_csv("C:\\Users\\admin\\Desktop\\Mall_Customers.csv")
# mydata.head()
x=mydata.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
km=KMeans(n_clusters=4)
km.fit(x)
pred=km.predict(x)
pred

plt.scatter(x[pred==0,0],x[pred==0,1],c='blue',label='cluster 1')
plt.scatter(x[pred==1,0],x[pred==1,1],c='red',label='cluster 2')
plt.scatter(x[pred==2,0],x[pred==2,1],c='green',label='cluster 3')
plt.scatter(x[pred==3,0],x[pred==3,1],c='yellow',label='cluster 4')
plt.legend()
plt.show()

#or method
plt.scatter(x[:,0],x[:,1],c=pred)
plt.show()
