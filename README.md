# Matrix-multiVector-product
## Scaricare matrici
Le matrici verranno salvate automaticamente nella cartella matrixFile. La prima istruzione fornisce i permessi di esecuzione al file .sh che scaricherà tutte le matrici.
```bash
chmod +x downloadMatrix.sh
./downloadMatrix.sh
```

NB: Eseguire dalla directory root

# How to build
```
$ make bin/debug --> builda una versione del progetto con informazioni di debug
$ make bin/release --> builda cercando di ottimizzare le prestazioni
```
Eseguire il programma senza parametri da linea di comando, ad esempio `$ bin/debug`.

 possibile cambiare alcuni parametri del programma direttamente nel codice. I commenti forniscono informazioni utili al riguardo.
