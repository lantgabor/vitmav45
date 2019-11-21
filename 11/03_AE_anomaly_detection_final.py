# -*- coding: utf-8 -*-
'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérejük
az alábbi szerzőt értesíteni.

2019 (c) Csapó Tamás Gábor (csapot kukac tmit pont bme pont hu)


# adatok eredetileg innen:
#   https://www.kaggle.com/mlg-ulb/creditcardfraud/data
# források:
#   https://hub.packtpub.com/using-autoencoders-for-detecting-credit-card-fraud-tutorial/
#   https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd
#   https://towardsdatascience.com/anomaly-detection-with-autoencoder-b4cdce4866a6
'''

# !wget smartlab.tmit.bme.hu/csapo/dl/creditcard.csv

# !ls -al

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
import matplotlib.pyplot as plt
import numpy as np

# adatok beolvasása pandas dataframe-be
df = pd.read_csv('creditcard.csv')

# először nézzük át, mi van az adatokban!

df.shape

# V1...V28: tömörített, anonimizált adat
# Time: időbélyeg
# Amount: összeg
# Class: osztály (0: normál, 1: csalás)
df

# van-e benne hiányzó érték
df.isnull().values.any()

# normál / csalás tranzakciók aránya
frauds = df[df.Class == 1]
normal = df[df.Class == 0]
print('normál:', normal.shape, 'csalás:', frauds.shape)

# mennyire különböző a normál / csalás tranzakciók összege?
frauds.Amount.describe()

normal.Amount.describe()

# mennyire különböző a normál / csalás tranzakciók összege?  - hisztogramon ábrázolva
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,5))

ax1.hist(frauds.Amount, bins = 20)
ax1.set_title('Csalás')

ax2.hist(normal.Amount, bins = 50)
ax2.set_title('Normál')
plt.xlabel('Összeg ($)')
plt.ylabel('Tranzakciók száma')
plt.xlim((0, 20000))
plt.yscale('log')

# csalás tranzakciók: látszik valamilyen mintázat időben?
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,5))

ax1.scatter(frauds.Time, frauds.Amount)
ax1.set_title('Csalás')

ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normál')

plt.xlabel('Idő (s)')
plt.ylabel('Összeg')

# adatok átrendezése: időtől nem függ
df = df.drop('Time',axis=1)

X = df.drop('Class',axis=1).values
y = df['Class'].values

X -= X.min(axis=0)
X /= X.max(axis=0)

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.1)

X_train.shape

data_in = Input(shape=(29,))
encoded = Dense(2,activation='tanh')(data_in)
decoded = Dense(29,activation='sigmoid')(encoded)
autoencoder = Model(data_in,decoded)

autoencoder.compile(optimizer='adam',loss='mean_squared_error')

autoencoder.fit(X_train, X_train, epochs = 5, batch_size=128, validation_data=(X_test,X_test))

pred = autoencoder.predict(X_test[0:10])

fig = plt.figure(figsize=(10,7))
plt.plot(pred[9], 'r')
plt.plot(X_test[9], 'b')
plt.legend(['predicted', 'true'])

encoder = Model(data_in,encoded)
enc = encoder.predict(X_test)

enc[0]

# a 'csalás' (piros) adatok valóban elkülönülnek a 'normál' adatoktól a 2D reprezentáció során

fig = plt.figure(figsize=(15,10))
scatter =plt.scatter(enc[:,0],enc[:,1],c=y_test, cmap='coolwarm', s=5.0)
scatter.axes.get_xaxis().set_visible(False)
scatter.axes.get_yaxis().set_visible(False)