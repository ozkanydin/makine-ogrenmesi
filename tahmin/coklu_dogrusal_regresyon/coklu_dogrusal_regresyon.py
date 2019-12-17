#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')



#veri on isleme
boy = veriler[['boy']]


boykilo = veriler[['boy','kilo']]
boykiloyas = veriler[['boy','kilo','yas']]



#encoder:  Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()


#cinsiyet kolonunu nümerikleştirdik ama dummy variable tuzağına düştüğümüz için tek kolununu aldık dataframe dönüşünde
#sadece labelencoder dönüşümü de yeterli olur 0 1 dönüşümü yapar
cins = veriler.iloc[:,-1:].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cins[:,0] = le.fit_transform(cins[:,0])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
cins=ohe.fit_transform(cins).toarray()

print('cins',cins)
#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )

sonuc2 =pd.DataFrame(data = boykiloyas, index = range(22), columns = ['boy','kilo','yas'])





sonuc3 = pd.DataFrame(data = cins[:,:1] , index=range(22), columns=['cinsiyet'])


#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi

#train ve teste bölme
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#-----------y_test yerine artık boy kolunu gelir.
boy = s2.iloc[:,3:4].values

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)


r2 = LinearRegression()
r2.fit(x_train,y_train)
y_pred = r2.predict(x_test)

#modelin ve modeldeki değişkenlerin başarısını ölçebileceğimiz bir yapı
import statsmodels.api as sm


#coklu dogrusal regresyon formülünde beta0 a benzetmeye çalışmak için 22 satır 1 kolonluk 1 ekliyoruz
#formül y = b0 + b1x1 + b2x2 + b3x3 + E
#axis 1 , kolon olarak ekler
x  = np.append(arr = np.ones((22,1)).astype(int), values=veri,axis=1 )

#herbir kolonu ifade edecek liste oluşturuyoruz
x_l = veri.iloc[:,[0,1,2,3,4,5]].values

#ols istatiksel değerleri çıkartmaya yarıyor bulmak istenilen değer boy (endog) bağlıntı kurduğumuz x_l (exog)
r =  sm.OLS(boy,x_l).fit()


print(r.summary())
#p value değerlerine göre geri eleme yöntemi kullanılır.





