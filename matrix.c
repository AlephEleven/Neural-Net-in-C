#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define ARRAY_BUFFER 4096
#define MAT_2D 2

#define NORM_MEAN 0
#define NORM_STDEV 1


typedef struct matrix {
    float* data;
    int* shape;
    int shape_len;
    int size;
} matrix;



#define GEN_MATRIX(DATA_NAME, SHAPE_NAME, MAT_NAME, MAT_SHAPEX, MAT_SHAPEY) float DATA_NAME[ARRAY_BUFFER];\
                             int SHAPE_NAME[2] = {MAT_SHAPEX, MAT_SHAPEY}; \
                             matrix MAT_NAME = {.data=DATA_NAME, .shape=SHAPE_NAME}; \
                             update_mat(&MAT_NAME, MAT_2D);

/*

Randomizers + Inits

*/

float random_uniform() { 
    return 2*((float)rand() / (float)RAND_MAX)-1; 
}

float random_gaussian(float mean, float stdev){
    float s = -1;
    float x;
    float y;

    while(s >= 1 || s < 0){
        x = random_uniform();
        y = random_uniform();
        s = x*x + y*y;
    }

    return mean + stdev*(x*sqrt((-2*log(s))/s));
}

void* mat_randn(matrix mat){
    for(int i = 0; i < mat.size; i++){
        mat.data[i] = random_gaussian(NORM_MEAN, NORM_STDEV);
    }
}

void* mat_ones(matrix mat){
    for(int i = 0; i < mat.size; i++){
        mat.data[i] = 1;
    }  
}

void* mat_zeros(matrix mat){
    for(int i = 0; i < mat.size; i++){
        mat.data[i] = 0;
    }  
}

void* update_mat(matrix* mat, int shape_len){
    int mat_size = 1;

    for(int i = 0; i < shape_len; i++){
        mat_size *= mat->shape[i];
    }

    mat->shape_len = shape_len;
    mat->size = mat_size;
}


/*

Index + Printing

*/

int index_2d(int x, int y, int* shape){
    return y + x*shape[1];
}


void* mat_getsubmatrix(matrix matA, matrix matB, int rowStart, int rowEnd, int colStart, int colEnd){
    int indB = 0;
    for(int i = rowStart; i < rowEnd; i++){
        for(int j = colStart; j < colEnd; j++){
            int indA = index_2d(i, j, matA.shape);
            matB.data[indB] = matA.data[indA]; 
            indB++;
        }
    }
}

void print_matrix(matrix mat){
    for(int i = 0; i < mat.shape[0]; i++){
        for(int j = 0; j < mat.shape[1]; j++){
            int ind = index_2d(i, j, mat.shape);
            printf("%f ", mat.data[ind]); 
        }
        printf("\n");
    }
}

void info_matrix(matrix mat){
    printf("shape=(%d, %d), size=%d, dim=%d\n", mat.shape[0], mat.shape[1], mat.size, mat.shape_len);
}

/*

Math Operations

*/

float mat_sum(matrix mat){
    float sum = 0;
    for(int i = 0; i < mat.shape[0]*mat.shape[1]; i++){
        sum += mat.data[i];
    }

    return sum;
}

float mat_prod(matrix mat){
    float sum = 1;
    for(int i = 0; i < mat.shape[0]*mat.shape[1]; i++){
        sum *= mat.data[i];
    }

    return sum;
}

void* mat_transpose(matrix matA, matrix matB){
    for(int i = 0; i < matA.shape[1]; i++){
        for(int j = 0; j < matA.shape[0]; j++){
            int indA = index_2d(j, i, matA.shape);
            int indB = index_2d(i, j, matB.shape);
            matB.data[indB] = matA.data[indA];
        }
    }  
}

float mat_sum_axis(matrix mat, int index, int axis){
    float sum = 0;
    if(axis == 0){
        for(int i = 0; i < mat.shape[1]; i++){
           int ind = index_2d(index, i, mat.shape);
           sum += mat.data[ind];
        }
        return sum;
    }
    if(axis == 1){
        for(int j = 0; j < mat.shape[0]; j++){
           int ind = index_2d(j, index, mat.shape);
           sum += mat.data[ind];
        }
        return sum;
    }
    return -1;
}

void* mat_mul(matrix matA, matrix matB, matrix matC){
    if(matA.shape[1] != matB.shape[0]){
        printf("WARNING! mat_mul: Shape A (%d x %d) not compatible with Shape B (%d x %d)\n", matA.shape[0], matA.shape[1], matB.shape[0], matB.shape[1]);
    }

    if(matC.shape[0] != matA.shape[0] || matC.shape[1] != matB.shape[1]){
        printf("WARNING! mat_mul: Shape C (%d x %d) not valid output shape for (%d x %d) * (%d x %d), Expected (%d x %d)\n", matC.shape[0], matC.shape[1], matA.shape[0], matA.shape[1], matB.shape[0], matB.shape[1], matA.shape[0], matB.shape[1]);
    }

    int row = matA.shape[0];
    int col = matB.shape[1];
    int inner = matA.shape[1];
    for(int i = 0; i < row; i++){
        for(int c = 0; c < inner; c++){
            
            for(int j = 0; j < col; j++){
                int indA = index_2d(i, c, matA.shape);
                int indB = index_2d(c, j, matB.shape);
                int indC = index_2d(i, j, matC.shape);
                matC.data[indC] += matA.data[indA]*matB.data[indB];
            }

        }
    }
}

void* mat_mul_shaped(matrix matA, matrix matB, matrix* matC){
    matC->shape[0] = matA.shape[0];
    matC->shape[1] = matB.shape[1];
    update_mat(matC, MAT_2D);

    mat_mul(matA, matB, *matC);
}

void* mat_add(matrix matA, matrix matB, matrix matC){
    for(int i = 0; i < matA.size; i++){
        matC.data[i] = matA.data[i]+matB.data[i];
    }
}

void* mat_sub(matrix matA, matrix matB, matrix matC){
    for(int i = 0; i < matA.size; i++){
        matC.data[i] = matA.data[i]-matB.data[i];
    }
}

void* mat_elementwise_mul(matrix matA, matrix matB, matrix matC){
    if(matA.shape[0] != matB.shape[0] || matA.shape[1] != matB.shape[1]){
        printf("WARNING! mat_elementwise_mul: Shape A (%d x %d) not compatible with Shape B (%d x %d)\n", matA.shape[0], matA.shape[1], matB.shape[0], matB.shape[1]);
    }


    for(int i = 0; i < matA.size; i++){
        matC.data[i] = matA.data[i]*matB.data[i];
    }
}

void* mat_elementwise_mul_shaped(matrix matA, matrix matB, matrix* matC){
    matC->shape[0] = matA.shape[0];
    matC->shape[1] = matA.shape[1];
    update_mat(matC, MAT_2D);

    mat_elementwise_mul(matA, matB, *matC);
}

void* mat_scale(matrix mat, float scale){
    for(int i = 0; i < mat.size; i++){
        mat.data[i] = mat.data[i]*scale;
    }
}

void* mat_apply(matrix mat, float (*f)(float)){
    for(int i = 0; i < mat.size; i++){
        mat.data[i] = f(mat.data[i]);
    }
}

void* mat_reshape(matrix* mat, int* new_shape){
    mat->shape = new_shape;
}

void* mat_content(matrix* mat, float* data){
    mat->data = data;
}

void* mat_copy(matrix* matA, matrix* matB){
    matB->data = matA->data;
}

void* mat_arange(matrix mat){
    for(int i = 0; i < mat.size; i++){
        mat.data[i] = i;
    }
}

void* mat_eye(matrix mat){
    int N = mat.shape[0];

    for(int i = 0; i < N; i++){
        mat.data[i*N + i] = 1;
    }
}

void* mat_diag_eye(matrix matA, matrix matB){
    int N = matB.shape[0];

    for(int i = 0; i < N; i++){
        matB.data[i*N + i] = matA.data[i];
    }
}

int mat_argmax(matrix A){
    int argmax = 0;
    float max_val = A.data[0];

    for(int i = 1; i < A.size; i++){
        if (A.data[i] > max_val){
            max_val = A.data[i];
            argmax = i;
        }
    }
    
    return argmax;
}


/*

Transforms

*/


void* mat_linear(matrix matW, matrix matx, matrix matb, matrix matz){
    /*
    z = Wx + b
    */
    mat_mul(matW, matx, matz);
    mat_add(matz, matb, matz);
}

void* mat_lineartransform(matrix matW, matrix matx, matrix matb, matrix matz, float (*f)(float)){
    mat_linear(matW, matx, matb, matz);
    mat_apply(matz, f);
}