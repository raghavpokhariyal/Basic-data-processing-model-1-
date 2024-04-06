import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

y= y.reshape(len(y),1)
print(y)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y= sc_y.fit_transform(y)    

print(x)
print(y)

from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf')
regressor.fit(x,y)

sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))

inv_x = sc_x.inverse_transform(x)
inv_y = sc_y.inverse_transform(y)

plt.scatter(inv_x, inv_y, color = 'red')
plt.plot(inv_x, sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

import numpy as np
import matplotlib.pyplot as plt


x_grid = np.arange(min(inv_x), max(inv_x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(inv_x, inv_y, color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# raghav pokhariyal
# B-23BCE11181

#yashwardhan sandilya
#B-23BCE11315