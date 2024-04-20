import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5])
y=np.array([2,3,4,5,6])

mx=np.mean(x)
my=np.mean(y)

N=np.sum((x-mx)*(y-my))
D=np.sum((x-mx)**2)

m=N/D

b = my - m*mx

pred = m*x + b

plt.scatter(x,y)
plt.plot(x,pred)
plt.show()

