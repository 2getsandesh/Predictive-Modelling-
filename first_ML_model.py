import pandas as pd

file_path="your_dataset.csv"    #save file path of the data set

data=pd.read_csv(file_path)    #read the data to a variable
data.describe()                #view the data table

data.columns           #to view the columns of the data (eg. Price)
y=data.Price           #Specify a Prediction target

features=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']   #create list of features
X=data[features]                #select data coresponding to the features

from sklearn.model_selection import train_test_split

#split the data set into train set and validation set to test the prediction
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)  

from sklearn.tree import DecisionTreeRegressor 

model=DecisionTreeRegressor(random_state=1)    
model.fit(train_X,train_y)                 #fit the model or train the model

from sklearn.metrics import mean_absolute_error

predicted_y = model.predict(val_X)     #Test the predictions using values other than train values

print(mean_absolute_error(val_y, predicted_y))    #print the error in prediction


# here you can do the same as a function
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(scores,key=scores.get)

# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)