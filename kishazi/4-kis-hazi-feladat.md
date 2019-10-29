# Deep Learning a gyakorlatban Python és LUA alapon
## Negyedik kis házi feladat

A cél egy előtanított CNN háló tovább tanítása:
* Válassz ki három tetszőleges kategóriát a Google Open Image Dataset-ből (GOID: https://storage.googleapis.com/openimages/web/index.html) és tölts le 600-600 képet kategóriánként.
* Tölts be tetszőleges előtanított hálót (pl. Inception v3, VGG, ResNet) és tanítsd tovább (transfer learning) a kiválasztott három kategóriával:
   * Előbb csak a végső fully-connected rétegek tanuljanak. 
   * Majd az így tovább tanított hálót tanítsátok még tovább úgy, hogy a konvolúciós rétegek felsőbb rétegei is tanulnak (az alsóbbak nem). 
* Egy elkülönített teszt adatbázison, a kiválasztott három kategóriával értékeld ki a megoldás hatékonyságát.
* Figyelj arra, hogy
   * a kép letöltés módszerét, scriptjét (ha saját kód) is mellékeld,
   * a tanító adatbázisban 400-400, a validációsban 100-100, a tesztben 100-100 kép legyen,
   * azonos előfeldolgozó eljárást használj, mint amit az előtanított hálónál alkalmaztak.

Tetszőleges deep learning keretrendszert használhatsz. A megoldást részletesen kommentezd és a tanítás kimenetét (log fájlját) is mellékeld. 
A leírás és kommentek angolul legyenek!

Segítségként javasoljuk az alábbi forrás felhasználását: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Beadási határido: 2019. november 19. 23:59

Beadás: http://smartlab.tmit.bme.hu/oktatas-deep-learning#kishazi 
