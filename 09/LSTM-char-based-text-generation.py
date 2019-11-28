# -*- coding: utf-8 -*-
'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérejük
az alábbi szerzőt értesíteni.

A kód elkészítéséhez az alábbi források kerültek felhasználásra:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

2019 (c) Gyires-Tóth Bálint (toth.b kukac tmit pont bme pont hu)
'''
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from urllib.request import urlretrieve
import unidecode
import numpy as np
import random
import sys
import re, cgi

# Szövegkorpusz kiválasztása és letöltése
url_book="http://mek.oszk.hu/00500/00501/00501.htm"
urlretrieve(url_book, 'book.html')
text = open("book.html", encoding='latin-1').read().lower()

# html tagek kiszedése a szövegkorpuszból
tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
no_tags = tag_re.sub('', text)
text = cgi.escape(no_tags) 

print('Karakterek száma a szövegben összesen:', len(text))

# Az előforduló karakterek megszámlálása
chars = sorted(list(set(text)))
print('Előforduló karakterek száma:', len(chars))

# Szótárban a karakter-szám és az inverz leképezés
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print ("Karakterleképezések:", indices_char)

# A szöveget maxlen hosszú egységekbe (ablakokba) tagoljuk. 
# Ezt most sentence-nek nevezzük, de ez nem egyezik a valós mondatokkal.
maxlen = 40
step = 3 # 3 karakterenként léptetjük az ablakot
sentences = []
next_chars = []

# a tanító adatok elkészítése maxlen ablakmérettel és step képésközzel
for i in range(0, len(text)-maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])
    
print('Tanítóminták száma:', len(sentences)) 
rand_ind = 2837
print('Egy random tanítóminta:', sentences[rand_ind], next_chars[rand_ind])

# a tanító adatok numerikus reprezentáckiójának elkészítése
print('Adatok tenzorba rendezése...')
X = np.zeros((len(sentences), maxlen, len(chars)))
y = np.zeros((len(sentences), len(chars)))

# A tanítóadatok one-hot kódolása
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence): 
        X[i,t,char_indices[char]] = 1
    y[i,char_indices[next_chars[i]]] = 1

print ("Tanító tenzor alakja:", X.shape)
print ("Teszt tenzor alakja:", y.shape)
print (X[rand_ind])
print (y[rand_ind])

# Mély tanuló modell létrehozása
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[-2], X.shape[-1])))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# mintavétel, ami újrasúlyozza a predikciót a temperature változó alapján 
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds) # Az összes lehetőség egyre szummázódjon (lásd softmax képlet)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas), preds

# kipróbáljuk a sample függvényt
fake_preds=[0.1, 0.2, 0.3, 0.15, 0.25]
for temp in [0.1, 0.5, 1, 2, 4]:
    print(fake_preds)
    proba, preds = sample(fake_preds,temp)
    print(preds)
    print(proba)
    
# tanítás és 10 epochonként szöveg generálása (összesen 10*10 epoch fut le)
start_index = random.randint(0, len(text) - maxlen - 1)
for iteration in range(1, 10):
    print()
    print('-' * 50)
    print('Iteráció', iteration)
    model.fit(X, y, batch_size=128, epochs=10)
    
    modelfile="model-it-"+str(iteration)+".h5" # menthetjük a modellt, hogy később tudjuk futtatni
    model.save(modelfile) # itt elmentjük a modellt, hogy később erre az állapotra vissza tudjunk könni

    for temp in [0.2, 0.5, 1.0, 1.2]: # a mintavételhez kell majd (temperature)
        print()
        print('----- hőmérséklet:', temp)
        generated = ''
        sentence = text[start_index: start_index + maxlen] # kiválasztunk egy kezdeti szöveget, amiből kiindulunk, a neuronháló ezt fogja folytatni
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400): # legenerálunk 400 karaktert egymás után
            # teszt adat one-hot kódolva LSTM bemenetére
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
            	x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0] # forward pass
            next_index,_ = sample(preds, temp) # kimeneten kapott eloszlásból mintát veszünk
            next_char = indices_char[next_index] # a mintát karakterre képezzük

            generated += next_char
            sentence = sentence[1:] + next_char # léptetünk egy karatert (első kiesik, utolsónak bejön a most generált)

            sys.stdout.write(next_char) # kiírjuk, amit generáltunk
            sys.stdout.flush()
       
        print()
