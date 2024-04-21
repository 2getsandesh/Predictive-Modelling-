import pandas as pd

df = pd.read_csv('details.csv')
print(df.head())

'''max_height = df['height'].max()
print(max_height)

max_height_person = df[df['height']==max_height]
print(df['height'].mean())
print(max_height_person['Age'])
'''

'''new_data = {'Name': ['Afthab'],
            'Age': [56],
            'height': [190]}

new_df = pd.DataFrame(new_data)
new_df.to_csv("details.csv", mode='a',index=False,header=False)
print(df.head())'''



