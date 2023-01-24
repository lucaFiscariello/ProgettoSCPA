#ifndef PRODUCT_H
#define PRODUCT_H

#include "matrix/matrix.h"

/**
 * @brief Registra tutte le informazioni utili al campionamento che identificano
 * le matrici che partecipano al prodotto matriciale.
 * 
 */
typedef struct matrix_sample_id{
    
    int numElements;    // NZ nel caso delle matrici sparse, k nel caso del multivettore
    int numBytes;       // grandezza in bytes della matrice in memoria
    char *name;         // nome identificativo della matrice

} MatrixSampleID;

/**
 * @brief Struttura che contiene i dati di un campione di misurazione
 * delle prestazioni di calcolo del prodotto matriciale. 
 */
typedef struct sample{

    /** Tempo di esecuzione del prodotto matriciale in secondi*/
    double execTimeSecs;

    /** Tempo di esecuzione del prodotto matriciale in nanosecondi*/
    double execTimeNsecs;
    
    /** Numero di miliardi di FLOating point OPerationS per secondo*/
    double gflops;
    
    /** Nome identifiativo dell'implementazione del prodotto matriciale*/
    char *productName;
    
    /** Numero di MBs elaborati per secondo*/
    double bandwidth;

    MatrixSampleID matrix1;

    MatrixSampleID matrix2;
    

} Sample;


/**
 * @brief
 * Funzione che implementa il prodotto matriciale in modo seriale. La funzione non è tenuta a conoscere il formato 
 * delle due matrici. L'unico vincolo ovviamente è che il numero di righe della prima matrice si 
 * uguale al numero di colonne della seconda matrice. 
 * 
 * @param matrix1
 * @param matrix2 
 * @param mResult Matrice in cui verrà scritto il risultato del prodotto matriciale
 * @param sample Dove devono essere scritti i dati di misurazione delle prestazioni
 * @return -1 se c'è stato un errore, 0 altrimenti 
 */
//int productMatrixMatrixSerial(Matrix *matrix1, Matrix *matrix2, Matrix *mResult, Sample *sample);

#endif