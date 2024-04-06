import matplotlib as mt 
import numpy as np 
import pandas as pd

dataset = pd.read_csv("50.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x= np.array(ct.fit_transform(x))

print(x)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_predict = regressor.predict(x_test)
np.set_printoptions(precision=2)
y_predict_reshaped = y_predict.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)
results = np.concatenate((y_predict_reshaped, y_test_reshaped), axis=1)
print(results)
    

# raghav pokhariyal
# B-23BCE11181

#yashwardhan sandilya
#B-23BCE11315