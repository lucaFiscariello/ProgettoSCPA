#ifndef PRODUCT_H
#define PRODUCT_H

#include "matrix/matrix.h"

/**
 * @brief Registra tutte le informazioni utili al campionamento che identificano
 * le matrici che partecipano al prodotto matriciale.
 * 
 */
typedef struct matrix_sample_id{
    
    long numElements;    // NZ nel caso delle matrici sparse, k nel caso del multivettore
    long numBytes;       // grandezza in bytes della matrice in memoria
    const char *name;         // nome identificativo della matrice
    const char *formatName;       // nome del formato della matrice  

} MatrixSampleID;

/**
 * Convenience constructor
*/
MatrixSampleID *newMatrixSampleID(long numElements, long numBytes, const char *name, const char *formatName);
void freeMatrixSampleID(MatrixSampleID *self);

/**
 * @brief Struttura che contiene i dati di un campione di misurazione
 * delle prestazioni di calcolo del prodotto matriciale. 
 */
typedef struct sample{

    /** parte in secondi del tempo di esecuzione del prodotto matriciale
     * Deve essere scritto dall'implementazione del prodotto matriciale
    */
    time_t execTimeSecs;

    /** parte in nanosecondi del tempo di esecuzione del prodotto matriciale
     * Deve essere scritto dall'implementazione del prodotto matriciale
    */
    long execTimeNsecs;
    
    /** Nome identifiativo dell'implementazione del prodotto matriciale.
     * deve essere scritto dall'implementazione del prodotto matriciale.
    */
    char *productName;

    /** Numero di miliardi di FLOating point OPerationS per secondo.
     * Può essere calcolato dall'experimenter.
    */
    double gflops;

    /** Numero di MBs elaborati per secondo.
     * Può essere calcolato dall'experimenter.
    */
    double bandwidth;

    /** trial index
     * Deve essere scritto dall'experimenter.
     */
    int trial;
    
    /** Dati identificativi della matrice 1.
     * Devono essere passati dal pilota dell'experimenter.
    */
    MatrixSampleID *m1SampleId;

    /** Dati identificativi della matrice 2.
     * Devono essere passati dal pilota dell'experimenter.
    */
    MatrixSampleID *m2SampleId;

} Sample;

/**
 * @brief sets the number of GFLOPS in the provided Sample.
 * @return il numero di GFLOPS calcolato
*/
double calcGflops(Sample *self);

/**
 * Use this function to set the bandwidth to the provided Sample.
 * @return il numero di MBs elaborati per secondo
*/
double calcBandwidth(Sample *self);

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

int productCsrMultivectorParallelCPU(Matrix *matrix1, Matrix *matrix2, Matrix *mResult, Sample *sample);

/**
 * Product between a sparse matrix in CSR format and a multivector
 * implemented in parallel on the GPU using CUDA.
*/
int productCsrMultivectorParallelGPU(Matrix *matrix1, Matrix *matrix2, Matrix *mResult, Sample *sample);

#endif