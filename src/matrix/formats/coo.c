#include "matrix/formats/coo.h"
#include "logger/logger.h"
#include <stdlib.h>
#include "dataStructures/ll.h"
#include <stdio.h>

double getCOO(Matrix *self, int r, int c){

    Node * row, *col, *element;
    DataCOO *data = (DataCOO *)self->data;

    for (int i = 0; i < self->numNonZero; i ++){
        ON_ERROR_LOG(getLL(data ->rows, i, &row), "couldn't get row %d\n", i);
        ON_ERROR_LOG(getLL(data ->cols, i, &col), "couldn't get col %d\n", i);
        if (*(int*)(row->value) == r && *(int*)(col->value) == c){
            ON_ERROR_LOG(getLL(data ->elements, i, &element) != 0, "couldn't get element in position %d\n", i);
            return *(double*)(element->value);
        }
    }
    return 0;
}

NotZeroElement* getNonZeroCOO(Matrix *self, int pos){

    NotZeroElement* nze = calloc(1, sizeof(NotZeroElement));

    Node * row, *col, *element;
    DataCOO *data = (DataCOO *)self->data;

    for (int i = 0; i < self->numNonZero; i ++){
        ON_ERROR_LOG(getLL(data ->rows, i, &row), "couldn't get row %d\n", i);
        ON_ERROR_LOG(getLL(data ->cols, i, &col), "%s: couldn't get col %d\n", i);
        ON_ERROR_LOG(getLL(data ->elements, i, &element), "couldn't get element in position %d\n", i);

        if (i==pos){
            nze->value=*(double*)(element->value);
            nze->col = *(int*)(col->value);
            nze->row= *(int*)(row->value);
            return nze ;
        }
    }

    return NULL;
}

int putCOO(Matrix *self, int r, int c, double val){

    DataCOO *data = (DataCOO *)self->data;
    Node *row, *col, *element;
    
    // holders for values to be stored in linked lists
    double *valHolder;
    int *colHolder, *rowHolder;

    // Ingrandiamo la dimensione della matrice se necessario.
    if (r >= self ->rows){
        self ->rows = r + 1;
    }
    if (c >= self ->cols){
        self ->cols = c + 1;
    }

    // Scorro tutti gli elementi della matrice cercando la coppia di indici (r,c)
    valHolder = malloc(sizeof(double));
    *valHolder = val;
    for (int i = 0; i < self ->numNonZero; i ++){
        
        ON_ERROR_LOG(getLL(data ->rows, i, &row), "couldn't get row %d\n", i);
        ON_ERROR_LOG(getLL(data ->cols, i, &col), "couldn't get col %d\n", i);

        if (*(int*)(row->value) == r && *(int*)(col->value) == c){
            
            ON_ERROR_LOG(getLL(data ->elements, i, &element) != 0, "couldn't get element in position %d\n", i); 
            
            // la riga e la colonna specificati sono già occupati da un elemento non zero, che andremo
            // a sostituire con il nuovo valore
            if (val != 0){
                element->value = (char*)valHolder;
                return 0;
            } else {
                // val è zero, quindi devo eliminare l'elemento e i suoi indici dalle liste
                free(valHolder);
                free(row ->value);
                free(col ->value);
                free(element ->value);
                ON_ERROR_LOG(popLL(&(data ->elements), i, NULL), "couldn't remove element %f in position %d\n", *(double*)(element->value), i);
                ON_ERROR_LOG(popLL(&(data ->rows), i, NULL), "couldn't remove row %d in position %d\n", *(int*)(row->value), i);
                ON_ERROR_LOG(popLL(&(data ->cols), i, NULL), ": couldn't remove col %d in position %d\n", *(int*)(col->value), i);
                self ->numNonZero --;
            }
        }
    }

    // Devo aggiungere elemento non-zero poiché non è stato trovato tra quelli già
    // presenti nella matrice

    if (val != 0){
        rowHolder = malloc(sizeof(int));
        *rowHolder = r;
        colHolder = malloc(sizeof(int));
        *colHolder = c;
        appendLL(&(data ->elements), valHolder);
        appendLL(&(data ->rows), rowHolder);
        appendLL(&(data ->cols), colHolder);
        self ->numNonZero ++;        
    }

    return 0;
}

void printCOO(Matrix *self){

    printf("elements: ");
    printLL(((DataCOO *)self->data)->elements, "%f", double);
    printf("rows: ");
    printLL(((DataCOO *)self->data)->rows, "%d", int);
    printf("cols: ");
    printLL(((DataCOO *)self->data)->cols, "%d", int);
}

Matrix *newMatrixCOO(){

    Matrix *self = newMatrix();
    DataCOO *data = calloc(1, sizeof(DataCOO));
    
    /** setup members*/
    self ->data = data;
    
    /** setup methods*/
    self ->put = putCOO;
    self ->get = getCOO;
    self ->getNonZero = getNonZeroCOO;
    self ->print = printCOO;

    return self;
}

void freeMatrixCOO(Matrix *self){
    
    DataCOO *data = (DataCOO *)self->data;
    
    // free all values of nodes
    Node *row, *col, *element;
    for(int i = 0; i < self ->numNonZero; i ++){
        ON_ERROR_LOG(getLL(data ->rows, i, &row), "%s: couldn't get row %d\n", i);
        ON_ERROR_LOG(getLL(data ->cols, i, &col), "%s: couldn't get col %d\n", i);
        ON_ERROR_LOG(getLL(data ->elements, i, &element), "%s: couldn't get element in position %d\n", i);
        free(row ->value);
        free(col->value);
        free(element ->value);
    }

    // destroy lists
    destroyLL(data ->rows);
    destroyLL(data ->cols);
    destroyLL(data ->elements);

    // free data
    free(data);

    // free matrix
    freeMatrix(self);
}

