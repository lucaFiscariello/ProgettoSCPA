#ifndef EXPERIMENTER_H
#define EXPERIMENTER_H

#include "matrix/matrix.h"
#include "product/product.h"

/**
 * Per ogni possibile esperimento (m1, m2, product) esegue
 * il prodotto matriciale e
 * registra le informazioni di prestazioni in un oggetto Sample, uno per ogni prova
 * eseguita, nell'array samples.
 * @param m1 array di matrici 1
 * @param msid1 array di informazioni utili al campionamento che identificano le matrici m1.
 * Devono essere nello stesso ordine e numero di m1.
 * @param numM1 numero di matrici m1
 * @param m2 array di matrici 2
 * @param msid2 array di informazioni utili al campionamento che identificano le matrici m2.
 * Devono essere nello stesso ordine e numero di m2.
 * @param numM2 numero di matrici m2
 * @param products array di puntatori alle funzioni che implementano il prodotto matriciale.
 * @param numProducts numero di prodotti matriciali
 * @param numTrials numero di volte per cui ogni esperimento deve essere ripetuto.
 * @param samples array multidimensionale in cui verranno scritti i puntatori ai Samples prodotti
 * da ogni
 * esperimento. Deve avere dimensione almeno pari a numM1 * numM2 * numProducts * numTrials.
 * I samples sono allocati dinamicamente, e l'ownership è trasferita al caller.
*/
int doExperiments(
    Matrix *m1[], MatrixSampleID *msid1[], int numM1,
    Matrix *m2[], MatrixSampleID *msid2[], int numM2,
    int (*products[])(Matrix *, Matrix *, Matrix *, Sample *), int numProducts,
    int numTrials,
    Sample *samples[]);

/**
 * Questa funzione permette di stampare su un csv tutti i samples ottenuti 
 * in seguito all'esecuzione dei prodotti matrice sparsa - multivettore
*/
int printSamplesToCSV(int numSamples, Sample *samples[], char *filename);


#endif // EXPERIMENTER_H