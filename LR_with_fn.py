import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x=np.array([1,2,3,4,5]).reshape(-1,1)
y=np.array([2,3,4,5,6])

model = LinearRegression()
model.fit(x,y)

predictions = model.predict(x)

plt.scatter(x,y,color="red",label="points")
plt.plot(x,predictions,color="Blue",label="Linear Regression")
plt.legend()
plt.show()