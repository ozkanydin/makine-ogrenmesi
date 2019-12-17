

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('ysaveri.csv')
#pd.read_csv("veriler.csv")

#veri on isleme
#verilerden ysanın ezber yapabileceği istenmeyen durumları çıkarttık.
X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

#encoder:  Kategorik -> Numeric

from sklearn.preprocessing import LabelEncoder
#ulkeleri ve cinsiyetleri label encoder ile dönüştürdük
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

#ulkede sadece 1 ve 0 kalması için onehotencoder yaptık.
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X = X[:,1:]


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection  import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#Yapay Sinir Ağı

import keras
from keras.models import Sequential
#sequantial bir ysa yapısının var olduğunu belirtir.
from keras.layers import Dense
#katman oluşturma

classifier = Sequential()

classifier.add(Dense(6, init = 'uniform', activation = 'relu' , input_dim = 11))
#11 giriş noronu ve 1 çıkış noronu var 6 gizli katmandaki noron sayısını ifade eder
#sinapsislerin ilklenmesi 0  a yakın bir değer olması init bölümü 
#activaction aktivasyon fonksiyonu bölümü relu kullanıldı 0 ın altında sıfır 
#sıfırın üstünde ise linear artan bir model 
#inputdim giriş katmanında kaç veri olduğu

#6 noronluk bir gizli katman daha ekledik
classifier.add(Dense(6, init = 'uniform', activation = 'relu'))

#cıkıs katmanının aktivasyon kodu sigmoid fonksiyonudur.
classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid'))


#ysa yı derleme

classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'] )

classifier.fit(X_train, y_train, epochs=50)
#epochs kaç adımda eğittik

#tahmin ettik
y_pred = classifier.predict(X_test)


#true false şeklii
y_pred = (y_pred > 0.5)


#confusion_matrix karşılaştırma yapar
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
#tahminler ile gerçek sonucu karşılaştırır.

print(cm)














