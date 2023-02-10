#include "experimenter/experimenter.h"
#include "matrix/formats/coo.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "logger/logger.h"
#include "libcsv/csv.h"

int doExperiments(
    Matrix *m1[], MatrixSampleID *msid1[], int numM1,
    Matrix *m2[], MatrixSampleID *msid2[], int numM2,
    int (*products[])(Matrix *, Matrix *, Matrix *, Sample *), int numProducts,
    int numTrials,
    Sample *samples[])
{
    
    // usiamo un buffer di matrici COO per memorizzare temporaneamente
    // i risultati dei prodotti matriciali, tanto non ci serve conservarli,
    // ci servono solo i campionamenti prestazionali
    Matrix *mrBuffer = NULL;
    int curSampleIndex;
        
    /**
     * experiment[i, j, p] --> (m1[i], m2[j], products[p])
    */
    for (int i = 0; i < numM1; i++){
        for (int j = 0; j < numM2; j++){
            for (int p = 0; p < numProducts; p++){

                // to do an experiment, we must cycle through all its trials
                for (int t = 0; t < numTrials; t ++){
                    
                    // reset mr buffer entry with pristine COO matrices
                    freeMatrixCOO(mrBuffer);                    
                    mrBuffer = newMatrixCOO();
                    
                    // initialize sample for the trial of this experiment
                    curSampleIndex = i * numM2 * numProducts * numTrials + j * numProducts * numTrials + p * numTrials + t;
                    samples[curSampleIndex] = calloc(1, sizeof(Sample));
                    samples[curSampleIndex] -> m1SampleId = msid1[i];
                    samples[curSampleIndex] -> m2SampleId = msid2[j];
                    samples[curSampleIndex] -> trial = t;

                    // do the trial of this experiment and calculate its perfomance.
                    ON_ERROR_LOG_AND_RETURN(products[p](m1[i], m2[j], mrBuffer, samples[curSampleIndex]), -1, "Error while doing trial %d for experiment %d, %d, %d\n", t, i, j, p);
                    calcGflops(samples[curSampleIndex]);
                    calcBandwidth(samples[curSampleIndex]);
                }
            }
        }
    }

    return 0;
}

int printSamplesToCSV(int numSamples, Sample *samples[], char *filename){

    FILE *csv;
    csv = fopen(filename,"wb");

    //Stampo header
    fprintf(csv,"execTimeSecs,");
    fprintf(csv,"execTimeNsecs,");
    fprintf(csv,"productName,");
    fprintf(csv,"gflops,");
    fprintf(csv,"bandwidth,");
    fprintf(csv,"numElements_mat1,");
    fprintf(csv,"numBytes_mat1,");
    fprintf(csv,"name_mat1,");
    fprintf(csv,"numElements_mat2,");
    fprintf(csv,"numBytes_mat2,");
    fprintf(csv,"name_mat2\n");
    
    //Stampo un sample per ogni riga
    for(int i=0; i< numSamples; i++){

        fprintf(csv,"%ld,",samples[i]->execTimeSecs);
        fprintf(csv,"%ld,",samples[i]->execTimeNsecs);
        fprintf(csv,"%s,",samples[i]->productName);
        fprintf(csv,"%f,",samples[i]->gflops);
        fprintf(csv,"%f,",samples[i]->bandwidth);
        fprintf(csv,"%ld,",samples[i]->m1SampleId->numElements);
        fprintf(csv,"%ld,",samples[i]->m1SampleId->numBytes);
        fprintf(csv,"%s,",samples[i]->m1SampleId->name);
        fprintf(csv,"%ld,",samples[i]->m2SampleId->numElements);
        fprintf(csv,"%ld,",samples[i]->m2SampleId->numBytes);
        fprintf(csv,"%s\n",samples[i]->m2SampleId->name);


    }



    return 0;

}

