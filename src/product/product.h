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

    /** parte in secondi del tempo di esecuzione del prodotto matriciale*/
    time_t execTimeSecs;

    /** parte in nanosecondi del tempo di esecuzione del prodotto matriciale*/
    long execTimeNsecs;
    
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
 * @param execTimeSecs
 * @param execTimeNsecs
 * @param numNonZero numero di elementi non nulli della matrice sparsa
 * @param nMVCols numero di colonne del multivettore
 * @return the number of GFLOPS
*/
double calcGflops(time_t execTimeSecs, long execTimeNsecs, int numNonZero, int nMVCols);

/**
 * Use this function to set the bandwidth to the provided Sample.
 * @param numMBytes numero di MBs elaborati, ad esempio la somma in MBs delle 
 * matrici del prodotto.
 * @param execTimeSecs
 * @param execTimeNsecs
 * @return the bandwidth in MB/s
*/
double calcBandwidth(double numMBytes, time_t execTimeSecs, long execTimeNsecs);


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
int productMatrixMatrixSerial(Matrix *matrix1, Matrix *matrix2, Matrix *mResult, Sample *sample);

/**
 * @brief
 * Funzione che implementa il prodotto matriciale in modo parallelo sulla GPU.
 * In particolare La funzione accetta in ingresso una matrice in formato Ellpack e un multivettore.
 * 
 * @param matrix1
 * @param matrix2 
 * @param mResult Matrice in cui verrà scritto il risultato del prodotto matriciale
 * @param sample Dove devono essere scritti i dati di misurazione delle prestazioni
 * @return -1 se c'è stato un errore, 0 altrimenti 
*/
int productMatrixMatrixParallelEllpack(Matrix *matrix1, Matrix *matrix2, Matrix *mResult, Sample *sample);

/**
 * Implementation of the product of a sparse matrix in ELLPACK format and a multivector
 * leveraging the parallelism of the multi-core CPU using OpenMP.
 * 
 * @param matrix1
 * @param matrix2 
 * @param mResult Matrice in cui verrà scritto il risultato del prodotto matriciale
 * @param sample Dove devono essere scritti i dati di misurazione delle prestazioni
 * @return -1 se c'è stato un errore, 0 altrimenti 
*/
int productEllpackMultivectorParallelCPU(Matrix *matrix1, Matrix *matrix2, Matrix *mResult, Sample *sample);

#endif