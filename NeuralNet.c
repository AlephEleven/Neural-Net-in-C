
#include "matrix.c"

#define GROUP_BUFFER 256
#define REFERENCE_BUFFER 256

#define GEN_DIFF_MATRIX(DATA_NAME, DATA_DX_NAME, \
                        SHAPE_NAME, DIFF_SHAPE_NAME, \
                        MAT_NAME, MAT_DX_NAME, DIFF_MAT_NAME, \
                        MAT_SHAPEX, MAT_SHAPEY, DIFF_MAT_SHAPEX, DIFF_MAT_SHAPEY) \
                            float DATA_NAME[ARRAY_BUFFER];\
                            float DATA_DX_NAME[ARRAY_BUFFER];\
                            int SHAPE_NAME[2] = {MAT_SHAPEX, MAT_SHAPEY}; \
                            int DIFF_SHAPE_NAME[2] = {DIFF_MAT_SHAPEX, DIFF_MAT_SHAPEY}; \
                            matrix MAT_NAME = {.data=DATA_NAME, .shape=SHAPE_NAME}; \
                            matrix MAT_DX_NAME = {.data=DATA_DX_NAME, .shape=DIFF_SHAPE_NAME}; \
                            update_mat(&MAT_NAME, MAT_2D); \
                            update_mat(&MAT_DX_NAME, MAT_2D); \
                            diff_matrix DIFF_MAT_NAME = {.mat = MAT_NAME, .mat_diff = MAT_DX_NAME};

#define Linear(input_size, output_size, \
                DATA_W, DATA_DX_W, matW_shape, matWdx_shape, matW, matWdx, diff_matW, \
                DATA_b, DATA_DX_b, matb_shape, matbdx_shape, matb, matbdx, diff_matb, \
                DATA_activation, matactivation_shape, matactivation, \
                LEARNABLE_PARAMETER_NAME) \
                GEN_DIFF_MATRIX(DATA_W, DATA_DX_W, matW_shape, matWdx_shape, matW, matWdx, diff_matW, output_size, input_size, 1, input_size); \
                mat_randn(matW); \
                GEN_DIFF_MATRIX(DATA_b, DATA_DX_b, matb_shape, matbdx_shape, matb, matbdx, diff_matb, output_size, 1, output_size, 1); \
                mat_zeros(matb); \
                GEN_MATRIX(DATA_activation, matactivation_shape, matactivation, input_size, output_size); \
                LearnableParameter LEARNABLE_PARAMETER_NAME = {.W = diff_matW, .b = diff_matb, .diff_activation = matactivation, .no_diff = 0};

#define Activation(DATA_ACTIVATION, ACTIVATION_SHAPE, matACTIVATION, ACTIVATION_SHAPEX, ACTIVATION_NAME) \
    GEN_MATRIX(DATA_ACTIVATION, ACTIVATION_SHAPE, matACTIVATION, ACTIVATION_SHAPEX, 1); \
    ActivationParameter ACTIVATION_NAME = {.diff_activation = matACTIVATION};

#define LossFunction(DATA_LOSS, LOSS_SHAPE, matLOSS, LOSS_SHAPEX, LOSS_FUNCTION, LOSS_NAME) \
    GEN_MATRIX(DATA_LOSS, LOSS_SHAPE, matLOSS, LOSS_SHAPEX, 1); \
    LossParameter LOSS_NAME = {.diff_activation = matLOSS, .loss_fn = LOSS_FUNCTION};

typedef struct diff_matrix {
    matrix mat;
    matrix mat_diff;
} diff_matrix;

typedef struct LearnableParameter {
    diff_matrix W;
    diff_matrix b;
    matrix diff_activation;
    int no_diff;
} LearnableParameter;

typedef struct ActivationParameter {
    matrix diff_activation;
} ActivationParameter;

typedef struct LossParameter {
    float loss;
    matrix diff_activation;
    float (*loss_fn)(matrix, matrix, matrix);
} LossParameter;

typedef struct ReferenceVariable {
    matrix* diff_activations;
    LearnableParameter* updates;
    int num_refs;
} ReferenceVariable;

typedef struct GroupedParameters {
    LearnableParameter *lp;
    ActivationParameter *ap;
    LossParameter llp;

    int lp_size;
    int ap_size;  
} GroupedParameters;

typedef struct Dataset {
    matrix X;
    matrix y;

    int in_shape;
    int out_shape;
    int N_samples;
} Dataset;


#define ForwardLinear(LINEAR_PARAMETER, INPUT_X, DATA_OUTPUT_Z, OUTPUT_Z_SHAPE, OUTPUT_Z, OUTPUT_Z_SHAPEX) \
                    GEN_MATRIX(DATA_OUTPUT_Z, OUTPUT_Z_SHAPE, OUTPUT_Z, OUTPUT_Z_SHAPEX, 1);\
                    diff_mat_linear(LINEAR_PARAMETER.W, INPUT_X, LINEAR_PARAMETER.b, OUTPUT_Z, LINEAR_PARAMETER.diff_activation);


void* diff_mat_linear(diff_matrix matW, matrix matx, diff_matrix matb, matrix matz, matrix diff_activation){
    /*
    
    z = Wx + b

    chain = diff_activation @ chain

    */

    mat_linear(matW.mat, matx, matb.mat, matz);

    //UPDATE W, dz/dW = x.T

    mat_transpose(matx, matW.mat_diff);

    //UPDATE b, dz/db = I

    mat_ones(matb.mat_diff);

    //UPDATE diff activation = W.T

    mat_transpose(matW.mat, diff_activation);
}


float sigmoid(float z){
    return 1/(1 + exp(-z));
}

#define ForwardSigmoid(ACTIVATION_PARAMETER, OUTPUT_Z) \
                    diff_sigmoid(OUTPUT_Z,  ACTIVATION_PARAMETER.diff_activation);

void* diff_sigmoid(matrix math, matrix diff_activation){
    /*
    h = sigmoid(z)

    diff activation = h*(1-h)    
    */

    mat_apply(math, sigmoid);

    for(int i = 0; i < diff_activation.size; i++){
        diff_activation.data[i] = math.data[i]*(1 - math.data[i]);
    }

}

float MSL(matrix matypred, matrix maty){
    float msl = 0;
    float diff = 0;
    for(int i = 0; i < matypred.size; i++){
        diff = maty.data[i]-matypred.data[i];
        msl += diff*diff;
    }

    return msl/matypred.size;
}

float diff_MSL(matrix matypred, matrix maty, matrix diff_activation){
    float loss = MSL(matypred, maty);
    float msl_mult = (-2.0/(float)matypred.size);

    for(int i = 0; i < diff_activation.size; i++){
        diff_activation.data[i] = msl_mult * (maty.data[i]-matypred.data[i]);
    }

    return loss;

}


matrix NetworkForward(matrix x, GroupedParameters network_params){

    ForwardLinear(network_params.lp[0], x, output1_data, output1_shape, output1, 20);
    ForwardSigmoid(network_params.ap[0], output1);

    ForwardLinear(network_params.lp[1], output1, output2_data, output2_shape, output2, 3);
    ForwardSigmoid(network_params.ap[1], output2);

    return output2;
}

void* NetworkBackward(ReferenceVariable refs, LossParameter lp, float lr){
    GEN_MATRIX(chains, chains_shape, matchains, lp.diff_activation.shape[0], lp.diff_activation.shape[1]);
    mat_content(&matchains, lp.diff_activation.data);

    GEN_MATRIX(tempchains, tempchains_shape, mattempchains, lp.diff_activation.shape[0], lp.diff_activation.shape[1]);

    GEN_MATRIX(dW, dW_shape, matdW, 0, 0);
    GEN_MATRIX(db, db_shape, matdb, 0, 0);

    int N = refs.num_refs-1;
    for(int i = N-1; i >= 0; i--){
    
        if(!refs.updates[i].no_diff){
            //gradient descent

            // W
            mat_mul_shaped(matchains, refs.updates[i].W.mat_diff, &matdW);

            mat_scale(matdW, lr);
            mat_sub(refs.updates[i].W.mat, matdW, refs.updates[i].W.mat);
            //printf("mul2\n");
            // b
            mat_elementwise_mul_shaped(matchains, refs.updates[i].b.mat_diff, &matdb);
            mat_scale(matdb, lr);
            mat_sub(refs.updates[i].b.mat, matdb, refs.updates[i].b.mat);
            //printf("mul3\n");

            mat_mul_shaped(refs.diff_activations[i], matchains, &mattempchains);
            matchains.shape[0] = refs.diff_activations[i].shape[0];
            mat_content(&matchains, mattempchains.data);


        }
        else{
            mat_elementwise_mul(refs.diff_activations[i], matchains, matchains);
        }
    }



}

void* NetworkTrain(Dataset dataset, GroupedParameters network_params, ReferenceVariable refs, LossParameter lp, int epochs, float lr){
    
    GEN_MATRIX(ypred, ypred_shape, matypred, dataset.out_shape, 1);

    GEN_MATRIX(data, data_shape, matdata, dataset.in_shape, 1);

    GEN_MATRIX(label, label_shape, matlabel, dataset.out_shape, 1);

    int acc = 0;

    int debug_time = (epochs/10 < 1) ? 1 :  epochs/10;

    for(int epoch = 1; epoch < epochs+1; epoch++){
        for(int ind = 0; ind < dataset.N_samples; ind++){
            //good
            mat_getsubmatrix(dataset.X, matdata, ind, ind+1, 0, dataset.in_shape);
            mat_getsubmatrix(dataset.y, matlabel, ind, ind+1, 0, dataset.out_shape);


            matypred = NetworkForward(matdata, network_params);

            acc += (mat_argmax(matypred) == mat_argmax(matlabel));
            

            lp.loss += lp.loss_fn(matypred, matlabel, lp.diff_activation);

            NetworkBackward(refs, lp, lr);
        }

        if(epoch%debug_time == 0){
            printf("Epoch %d: Loss = %f | Accuracy = %d/%d\n", epoch, lp.loss/dataset.N_samples, acc, dataset.N_samples);
        }
        lp.loss = 0;
        acc = 0;
    }


}


#define TOTAL_LAYERS 2
#define TOTAL_ACTIVATIONS 2
#define TOTAL_PASSES 4

int main(int argc, char** argv){
    srand(time(NULL));
    LearnableParameter skip; skip.no_diff = 1;

    //LAYERS
    Linear(4, 20, W1, W1dx, matW1_shape, matW1dx_shape, matW1, matW1dx, diff_matW1,
                 b1, b1dx, matb1_shape, matb1dx_shape, matb1, matb1dx, diff_matb1,
                 activation1, matactivation1_shape, matactivation1, fc1);

    Linear(20, 3, W2, W2dx, matW2_shape, matW2dx_shape, matW2, matW2dx, diff_matW2,
                 b2, b2dx, matb2_shape, matb2dx_shape, matb2, matb2dx, diff_matb2,
                 activation2, matactivation2_shape, matactivation2, fc2);

    //ACTIVATIONS
    Activation(sig1, sig1_shape, matsig1, 20, Sigmoid1);

    Activation(sig2, sig2_shape, matsig2, 3, Sigmoid2);

    //LOSS
    LossFunction(msl, msl_shape, matmsl, 3, diff_MSL, MeanSquaredLoss);

    //package network to reference variable

    LearnableParameter fcs[TOTAL_LAYERS] = {fc1, fc2};
    ActivationParameter activations[TOTAL_ACTIVATIONS] = {Sigmoid1, Sigmoid2};

    GroupedParameters network_params = {fcs, activations, MeanSquaredLoss, TOTAL_LAYERS, TOTAL_ACTIVATIONS};

    matrix diff_activations[TOTAL_PASSES] = {fc1.diff_activation, 
                                             Sigmoid1.diff_activation,
                                             fc2.diff_activation,
                                             Sigmoid2.diff_activation};
    
    LearnableParameter update_params[TOTAL_PASSES] = {fc1, 
                                                      skip,
                                                      fc2,
                                                      skip};

    ReferenceVariable references = {.diff_activations = diff_activations, .updates=update_params, .num_refs = TOTAL_PASSES};



    float X[ARRAY_BUFFER] = {5.1, 3.5, 1.4, 0.2, 4.9, 3. , 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6,
       3.1, 1.5, 0.2, 5. , 3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4,
       1.4, 0.3, 5. , 3.4, 1.5, 0.2, 4.4, 2.9, 1.4, 0.2, 4.9, 3.1, 1.5,
       0.1, 5.4, 3.7, 1.5, 0.2, 4.8, 3.4, 1.6, 0.2, 4.8, 3. , 1.4, 0.1,
       4.3, 3. , 1.1, 0.1, 5.8, 4. , 1.2, 0.2, 5.7, 4.4, 1.5, 0.4, 5.4,
       3.9, 1.3, 0.4, 5.1, 3.5, 1.4, 0.3, 5.7, 3.8, 1.7, 0.3, 5.1, 3.8,
       1.5, 0.3, 5.4, 3.4, 1.7, 0.2, 5.1, 3.7, 1.5, 0.4, 4.6, 3.6, 1. ,
       0.2, 5.1, 3.3, 1.7, 0.5, 4.8, 3.4, 1.9, 0.2, 5. , 3. , 1.6, 0.2,
       5. , 3.4, 1.6, 0.4, 5.2, 3.5, 1.5, 0.2, 5.2, 3.4, 1.4, 0.2, 4.7,
       3.2, 1.6, 0.2, 4.8, 3.1, 1.6, 0.2, 5.4, 3.4, 1.5, 0.4, 5.2, 4.1,
       1.5, 0.1, 5.5, 4.2, 1.4, 0.2, 4.9, 3.1, 1.5, 0.2, 5. , 3.2, 1.2,
       0.2, 5.5, 3.5, 1.3, 0.2, 4.9, 3.6, 1.4, 0.1, 4.4, 3. , 1.3, 0.2,
       5.1, 3.4, 1.5, 0.2, 5. , 3.5, 1.3, 0.3, 4.5, 2.3, 1.3, 0.3, 4.4,
       3.2, 1.3, 0.2, 5. , 3.5, 1.6, 0.6, 5.1, 3.8, 1.9, 0.4, 4.8, 3. ,
       1.4, 0.3, 5.1, 3.8, 1.6, 0.2, 4.6, 3.2, 1.4, 0.2, 5.3, 3.7, 1.5,
       0.2, 5. , 3.3, 1.4, 0.2, 7. , 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5,
       6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4. , 1.3, 6.5, 2.8, 4.6, 1.5, 5.7,
       2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1. , 6.6, 2.9,
       4.6, 1.3, 5.2, 2.7, 3.9, 1.4, 5. , 2. , 3.5, 1. , 5.9, 3. , 4.2,
       1.5, 6. , 2.2, 4. , 1. , 6.1, 2.9, 4.7, 1.4, 5.6, 2.9, 3.6, 1.3,
       6.7, 3.1, 4.4, 1.4, 5.6, 3. , 4.5, 1.5, 5.8, 2.7, 4.1, 1. , 6.2,
       2.2, 4.5, 1.5, 5.6, 2.5, 3.9, 1.1, 5.9, 3.2, 4.8, 1.8, 6.1, 2.8,
       4. , 1.3, 6.3, 2.5, 4.9, 1.5, 6.1, 2.8, 4.7, 1.2, 6.4, 2.9, 4.3,
       1.3, 6.6, 3. , 4.4, 1.4, 6.8, 2.8, 4.8, 1.4, 6.7, 3. , 5. , 1.7,
       6. , 2.9, 4.5, 1.5, 5.7, 2.6, 3.5, 1. , 5.5, 2.4, 3.8, 1.1, 5.5,
       2.4, 3.7, 1. , 5.8, 2.7, 3.9, 1.2, 6. , 2.7, 5.1, 1.6, 5.4, 3. ,
       4.5, 1.5, 6. , 3.4, 4.5, 1.6, 6.7, 3.1, 4.7, 1.5, 6.3, 2.3, 4.4,
       1.3, 5.6, 3. , 4.1, 1.3, 5.5, 2.5, 4. , 1.3, 5.5, 2.6, 4.4, 1.2,
       6.1, 3. , 4.6, 1.4, 5.8, 2.6, 4. , 1.2, 5. , 2.3, 3.3, 1. , 5.6,
       2.7, 4.2, 1.3, 5.7, 3. , 4.2, 1.2, 5.7, 2.9, 4.2, 1.3, 6.2, 2.9,
       4.3, 1.3, 5.1, 2.5, 3. , 1.1, 5.7, 2.8, 4.1, 1.3, 6.3, 3.3, 6. ,
       2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3. , 5.9, 2.1, 6.3, 2.9, 5.6, 1.8,
       6.5, 3. , 5.8, 2.2, 7.6, 3. , 6.6, 2.1, 4.9, 2.5, 4.5, 1.7, 7.3,
       2.9, 6.3, 1.8, 6.7, 2.5, 5.8, 1.8, 7.2, 3.6, 6.1, 2.5, 6.5, 3.2,
       5.1, 2. , 6.4, 2.7, 5.3, 1.9, 6.8, 3. , 5.5, 2.1, 5.7, 2.5, 5. ,
       2. , 5.8, 2.8, 5.1, 2.4, 6.4, 3.2, 5.3, 2.3, 6.5, 3. , 5.5, 1.8,
       7.7, 3.8, 6.7, 2.2, 7.7, 2.6, 6.9, 2.3, 6. , 2.2, 5. , 1.5, 6.9,
       3.2, 5.7, 2.3, 5.6, 2.8, 4.9, 2. , 7.7, 2.8, 6.7, 2. , 6.3, 2.7,
       4.9, 1.8, 6.7, 3.3, 5.7, 2.1, 7.2, 3.2, 6. , 1.8, 6.2, 2.8, 4.8,
       1.8, 6.1, 3. , 4.9, 1.8, 6.4, 2.8, 5.6, 2.1, 7.2, 3. , 5.8, 1.6,
       7.4, 2.8, 6.1, 1.9, 7.9, 3.8, 6.4, 2. , 6.4, 2.8, 5.6, 2.2, 6.3,
       2.8, 5.1, 1.5, 6.1, 2.6, 5.6, 1.4, 7.7, 3. , 6.1, 2.3, 6.3, 3.4,
       5.6, 2.4, 6.4, 3.1, 5.5, 1.8, 6. , 3. , 4.8, 1.8, 6.9, 3.1, 5.4,
       2.1, 6.7, 3.1, 5.6, 2.4, 6.9, 3.1, 5.1, 2.3, 5.8, 2.7, 5.1, 1.9,
       6.8, 3.2, 5.9, 2.3, 6.7, 3.3, 5.7, 2.5, 6.7, 3. , 5.2, 2.3, 6.3,
       2.5, 5. , 1.9, 6.5, 3. , 5.2, 2. , 6.2, 3.4, 5.4, 2.3, 5.9, 3. ,
       5.1, 1.8};

    int X_shape[2] = {150, 4};
    matrix matX = {.data=X, .shape=X_shape};
    update_mat(&matX, MAT_2D);

    float y[ARRAY_BUFFER] = {1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
       0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
       0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
       1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
       0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
       0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
       1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
       0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
       0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0.,
       0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
       0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
       1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
       0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
       0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
       1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
       0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
       0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
       1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
       0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
       1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
       0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
       0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
       1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
       0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
       0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
       1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
       0., 1., 0., 0., 1., 0., 0., 1};
    int y_shape[2] = {150, 3};
    matrix maty = {.data=y, .shape=y_shape};
    update_mat(&maty, MAT_2D);

    Dataset data_train = {.X = matX,
                          .y = maty,
                          .in_shape = 4,
                          .out_shape = 3,
                          .N_samples = 150};

    NetworkTrain(data_train, network_params, references, MeanSquaredLoss, 100, 0.01);



    
    return 0;
}