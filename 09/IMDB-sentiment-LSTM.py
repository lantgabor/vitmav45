# -*- coding: utf-8 -*-
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# paraméterek:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
hidden_dims = 250
epochs = 2

print('Adatok betöltése...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'tanító adat')
print(len(x_test), 'teszt adat')

print('A minták maxlen hosszúra történő kiegészítése (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Modell felépítése...')
model = Sequential()

# A "vocab" indexek beágyazásával kezdjük
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# a Convolution1D réteg célja, hogy betanulja az egymás utáni szócsoportokat
model.add(LSTM(64))
				 
# és sima előre csatolt réteg
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# bináris osztályozás - pozitív vagy negatív szentiment
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
			  
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
		  
print(model.evaluate(x_test, y_test))