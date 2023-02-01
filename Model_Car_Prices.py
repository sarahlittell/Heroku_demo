import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

Carprices=pd.read_csv('cars.csv')
X=Carprices[['Year','Miles']]
y=Carprices['Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

lm=LinearRegression()
lm.fit(X_train,y_train)

pickle.dump(lm, open('Car_Prices_model.pickle','wb'))