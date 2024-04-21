import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

'''boston = load_boston()
x = boston.data
y = boston.target'''

boston = pd.read_csv('HousingData.csv')
x=boston.drop(columns=['MEDV'],axis=1)
y=boston["MEDV"]

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=42)

DT = DecisionTreeRegressor(random_state=42,max_depth=4)
RF = RandomForestRegressor(n_estimators=100, random_state=42)

DT.fit(xtrain,ytrain)
RF.fit(xtrain,ytrain)

DTpred = DT.predict(xtest)
RFpred = RF.predict(xtest)

plt.scatter(ytest,ytest,color="red")
plt.scatter(ytest,DTpred,color="green")
plt.scatter(ytest,RFpred,color="blue")
plt.figure(figsize=(20,10))
plot_tree(DT,feature_names=boston["MEDV"])
plot_tree(RF.estimators_[0], feature_names=boston["MEDV"])
plt.show()