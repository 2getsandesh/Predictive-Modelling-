import pandas as pd

new_data = {'Name': ['Sandesh','Rajesh','Divya','Sarang'],
        'Age': [21,52,45,16],
        'height':[185,175,150,155]}

df = pd.DataFrame(new_data)

df.to_csv('details.csv', header=True, index=False)