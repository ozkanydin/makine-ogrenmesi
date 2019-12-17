import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
veriler = pd.read_csv('veriler.csv')
from sklearn.preprocessing import Imputer



#kategorik veriler

ulkeler = veriler.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder
#LabelEncoder herbir değer için sayısal bir değer koyar

le = LabelEncoder()
ulkeler[:,0] = le.fit_transform(ulkeler[:,0])
#hem değiştirip hem de uygular.(fit_transform)

print(ulkeler)
#her ulke için 0-2 arasında bir değer verdi(tr için 1 , us için 2 , fr için 0)


from sklearn.preprocessing import OneHotEncoder
#OneHotEncoder ile 3 ülke için üç sütün oluşturup satırda hangi ülke varsa o ülke sütünuna 1 diğerlerini 0 yapmamızda
#yardımcı olacak.
#--------------------------tr----------fr------------us
#örnek   tr                 1           0             0
ohe = OneHotEncoder(categorical_features="all")
ulkeler = ohe.fit_transform(ulkeler).toarray()
print(ulkeler)


