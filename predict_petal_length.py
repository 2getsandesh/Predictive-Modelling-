import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

x=df[['sepal length (cm)', 'species']]
y=df['petal length (cm)']

model=DecisionTreeRegressor(random_state=42)
model.fit(x,y)

x1=[[5.1, 2]]
pred=model.predict(x)

for actual,predicted in zip(y,pred):
    print(f"Actual: {actual} predicted: {predicted}")

