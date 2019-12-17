

# kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('satislar.csv')


satislar = veriler[['Satislar']]
aylar = veriler[['Aylar']]




#train ve teste bölme
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar, test_size=0.33, random_state=0)

'''
#standartlaştırma
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''
#doğrusal regresyonun model inşası
from sklearn.linear_model import  LinearRegression
lr = LinearRegression()

lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)

print(y_test)
print(tahmin)






