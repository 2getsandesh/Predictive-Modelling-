import pandas as pd

file_path="your_dataset.csv"    #save file path of the data set

data=pd.read_csv(file_path)    #read the data to a variable
data.describe()                #view the data table

data.columns           #to view the columns of the data (eg. Price)
y=data.Price           #Specify a Prediction target

features=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']   #create list of features
X=data[features]                #select data coresponding to the features

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)  #split the data set into train set and validation set to test the prediction

from sklearn.tree import DecisionTreeRegressor 

model=DecisionTreeRegressor(random_state=1)    
model.fit(train_X,train_y)                 #fit the model or train the model

from sklearn.metrics import mean_absolute_error

predicted_y = model.predict(val_X)     #Test the predictions using values other than train values

print(mean_absolute_error(val_y, predicted_y))    #print the error in prediction



