import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

x=df[['sepal length (cm)']]
y=df['petal length (cm)']

model=DecisionTreeRegressor(random_state=42)
model.fit(x,y)
x1=[[2.5]]
pred=model.predict(x1)
print(pred)
