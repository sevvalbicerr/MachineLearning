# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:57:26 2022

@author: sevva
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np

df=pd.read_csv("odev_tenis.txt")


"""humidity kolonu bizim bağımlı değişkenimiz olacak.
 label-encoder ile kategorik veriyi kullanılabilir hale getireceğiz."""
#Diğer kolonlar bağımsız değişkenimiz olacak
#Windy kolonu için yine label-encoder kullanacağız
#Outlook kolonunda one-hot-encoding kullanmalıyız.

df2=df.iloc[:,-2:].apply(LabelEncoder().fit_transform)

outlook=df.iloc[:,:1]
ohe = OneHotEncoder()
outlook_ohe=ohe.fit_transform(outlook).toarray()

#one-hot encoding uyguanan kolona ve df'a concat işlemi uygulayacağız

#bunun için önce havadurumu kolonunu dataframe hale getirmeliyiz
weather = pd.DataFrame(data = outlook_ohe, index = range(14), columns=['overcast','rainy','sunny'])
#artık concat işlemini gerçekleştirebiliriz.
l_df = pd.concat([weather,df.iloc[:,1:3]],axis = 1)
l_df = pd.concat([l_df,df2], axis = 1)

y=l_df.iloc[:,-1:]
X=l_df.iloc[:,:-1]

# Test ve train datasetlerini oluşturalım

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.33)

#model oluşturalım ve eğitelim

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)


# Değişken seçimi konusunda geriye doğru eleme yöntemini uygulayalım

""" Geriye doğru eleme yönteminde p value Sl değerden fazla olduğunda o değişkeni(genelde 0.05)
modelden çıkartarak iterasyonları devam ettirmek gerekiyor.
"""

X_l = l_df.iloc[:,[1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
r_ols = sm.OLS(endog = y, exog =X_l)
r = r_ols.fit()
print(r.summary())

#birinci kolonu eleyerek tekrardan sonuçları gözlemliyoruz

l_df = l_df.iloc[:,1:]
X_l = l_df.iloc[:,[1,2,3,4]].values
X_l=np.array(X_l,dtype=float)
r_ols = sm.OLS(endog = y, exog =X_l)
r = r_ols.fit()
print(r.summary())

l_df = l_df.iloc[:,:-1]
X_l = l_df.iloc[:,[1,2,3]].values
X_l=np.array(X_l,dtype=float)
r_ols = sm.OLS(endog = y, exog =X_l)
r = r_ols.fit()
print(r.summary())

#son durumda  R-squared (uncentered):0.714
