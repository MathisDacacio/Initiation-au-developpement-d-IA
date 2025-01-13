import pandas as pd
from sklearn.linear_model import LinearRegression

file = pd.read_csv("DataBase/final_scout_not_dummy.csv", sep=",")

logreg = LinearRegression()

learnData = file[['km']]
target = file['price']

print(learnData)

logreg.fit(learnData, target)

km_values = [[56013], [30000], [10000], [200000], [60000], [40000]]
result = logreg.predict(km_values)

print(result)