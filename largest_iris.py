from sklearn.datasets import load_iris
import pandas as pd

def largest(x):
    new_df = df[df['species'] == x]
    lar = new_df.max()
    lar.to_csv("Largest.csv")
    print(lar)
     

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]
#print(iris.target_names)

largest('setosa')
largest('versicolor')
largest('virginica')

