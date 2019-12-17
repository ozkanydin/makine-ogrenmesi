import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
eksikveriler = pd.read_csv('eksikveriler.csv')
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
#imputer ile eksik verileri belirtiyoruz ve ortalama stratejisini kullanıyoruz.

yas = eksikveriler.iloc[:,1:4].values
#pandas kütüphanesinin hangi kolonları almak istediğimizi belirten alt fonksiyon
#numerik olmayan kolonları almadık
imputer = imputer.fit(yas[:,1:4])
#ortalamayı aldık
yas[:,1:4] = imputer.transform(yas[:,1:4])
#eksik veriyi değiştirdik
print(yas)

