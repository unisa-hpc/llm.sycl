//
// Created by saleh on 19/05/24.
//

#pragma once

#include "core/Tensor.h"
#include "common/utils.h"
#include "common/common.h"
#include "kernels/Encoder.h"
#include "kernels/Residual.h"
#include "kernels/MatmulBias.h"
#include "kernels/Unpermute.h"
#include "kernels/Permute.h"
#include "kernels/Softmax.h"
#include "kernels/LayerNorm.h"
#include "kernels/Attention.h"
#include "kernels/Gelu.h"

#define NUM_PARAMETER_TENSORS 16
#define NUM_ACTIVATION_TENSORS 21

namespace llmsycl::model {
    class Model {
    private:
        /*************************************************************************************
         * Parameters:
         ***********************/
        int max_seq_len;                     ///< max sequence length, e.g. 1024
        int vocab_size;                      ///< vocab size, e.g. 50257
        int padded_vocab_size;               ///< padded to e.g. %128==0, 50304
        int num_layers;                      ///< number of layers, e.g. 12
        int num_heads;                       ///< number of heads in attention, e.g. 12
        int channels;                        ///< number of channels, e.g. 768
        size_t param_sizes[NUM_PARAMETER_TENSORS];
        core::TensorPtr<float> wte;          ///< (V, C)
        core::TensorPtr<float> wpe;          ///< (maxT, C)
        core::TensorPtr<float> ln1w;         ///< (L, C)
        core::TensorPtr<float> ln1b;         ///< (L, C)
        core::TensorPtr<float> qkvw;         ///< (L, 3*C, C)
        core::TensorPtr<float> qkvb;         ///< (L, 3*C)
        core::TensorPtr<float> attprojw;     ///< (L, C, C)
        core::TensorPtr<float> attprojb;     ///< (L, C)
        core::TensorPtr<float> ln2w;         ///< (L, C)
        core::TensorPtr<float> ln2b;         ///< (L, C)
        core::TensorPtr<float> fcw;          ///< (L, 4*C, C)
        core::TensorPtr<float> fcb;          ///< (L, 4*C)
        core::TensorPtr<float> fcprojw;      ///< (L, C, 4*C)
        core::TensorPtr<float> fcprojb;      ///< (L, C)
        core::TensorPtr<float> lnfw;         ///< (C)
        core::TensorPtr<float> lnfb;         ///< (C)
        int batch_size; // the batch size (B) of current forward pass
        int seq_len; // the sequence length (T) of current forward pass
        core::TensorPtr<int> inputs; // the input tokens for the current forward pass
        core::TensorPtr<int> targets; // the target tokens for the current forward pass
        size_t num_parameters;

        /*************************************************************************************
         * ACTIVATIONS:
         ***********************/
        core::TensorPtr<float> encoded;      ///< (B, T, C)
        core::TensorPtr<float> ln1;          ///< (L, B, T, C)
        core::TensorPtr<float> ln1_mean;     ///< (L, B, T)
        core::TensorPtr<float> ln1_rstd;     ///< (L, B, T)
        core::TensorPtr<float> atty;         ///< (L, B, T, C)
        core::TensorPtr<float> att;          ///< (L, B, NH, T, T)
        core::TensorPtr<float> attproj;      ///< (L, B, T, C)
        core::TensorPtr<float> residual2;    ///< (L, B, T, C)
        core::TensorPtr<float> ln2;          ///< (L, B, T, C)
        core::TensorPtr<float> ln2_mean;     ///< (L, B, T)
        core::TensorPtr<float> ln2_rstd;     ///< (L, B, T)
        core::TensorPtr<float> fch;          ///< (L, B, T, 4*C)
        core::TensorPtr<float> fch_gelu;     ///< (L, B, T, 4*C)
        core::TensorPtr<float> fcproj;       ///< (L, B, T, C)
        core::TensorPtr<float> residual3;    ///< (L, B, T, C)
        core::TensorPtr<float> lnf;          ///< (B, T, C)
        core::TensorPtr<float> lnf_mean;     ///< (B, T)
        core::TensorPtr<float> lnf_rstd;     ///< (B, T)

        core::TensorPtr<float> losses; // (B, T)
        // adding these two compared to the CPU .c code, needed for attention kernel as buffers
        core::TensorPtr<float> qkvr; // (L, B, T, 3*C)
        // in inference mode, this dBuff will store the logits
        // in training mode, this dBuff will contain the *gradients* of the logits.
        // during the processing of transformer blocks, we will also use this as a
        // general scratchpad dBuff. Allocation is made large enough to hold (B, T, 3C),
        // (B, NH, T, T), and (B, T, V) shaped tensors.
        core::TensorPtr<float> output;
        size_t act_sizes[NUM_ACTIVATION_TENSORS];
        size_t num_activations;
        bool isAllocated = false;


    public:

        void loadCheckpoint(const std::string &checkpointPath) {
            // read in model from a checkpoint file
            FILE *model_file = fopenCheck(checkpointPath.c_str(), "rb");
            int model_header[256];
            freadCheck(model_header, sizeof(int), 256, model_file);
            if (model_header[0] != 20240326) {
                logger->error("Bad magic model file");
                exit(EXIT_FAILURE);
            }
            if (model_header[1] != 3) {
                // was bumped from 1 -> 3 to incorporate the padded vocab size
                logger->error("Bad version in model file");
                logger->error("---> HINT: try to re-run `python train_gpt2.py`");

                exit(EXIT_FAILURE);
            }

            // read in hyperparameters
            max_seq_len = model_header[2];
            vocab_size = model_header[3];
            num_layers = model_header[4];
            num_heads = model_header[5];
            channels = model_header[6];
            padded_vocab_size = model_header[7];

            // allocate space for all the parameters and read them in
            {
                int Vp = padded_vocab_size;
                int C = channels;
                int maxT = max_seq_len;
                int L = num_layers;
                param_sizes[0] = Vp * C;                ///< wte
                param_sizes[1] = maxT * C;              ///< wpe
                param_sizes[2] = L * C;                 ///< ln1w
                param_sizes[3] = L * C;                 ///< ln1b
                param_sizes[4] = L * (3 * C) * C;       ///< qkvw
                param_sizes[5] = L * (3 * C);           ///< qkvb
                param_sizes[6] = L * C * C;             ///< attprojw
                param_sizes[7] = L * C;                 ///< attprojb
                param_sizes[8] = L * C;                 ///< ln2w
                param_sizes[9] = L * C;                 ///< ln2b
                param_sizes[10] = L * (4 * C) * C;      ///< fcw
                param_sizes[11] = L * (4 * C);          ///< fcb
                param_sizes[12] = L * C * (4 * C);      ///< fcprojw
                param_sizes[13] = L * C;                ///< fcprojb
                param_sizes[14] = C;                    ///< lnfw
                param_sizes[15] = C;                    ///< lnfb
            }

            // count the number of parameters
            num_parameters = 0;
            for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
                num_parameters += param_sizes[i];
            }

            // create memory for model parameters on the device
            {
                /*
                float **ptrs[] = {
                        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
                        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
                        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
                };
                */
                auto *params_memory_cpu = new float[num_parameters];
                freadCheck(params_memory_cpu, sizeof(float), num_parameters, model_file);
                size_t offsetAccumulator = 0;

                // 0
                wte = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[0]}),
                                                            params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[0];

                // 1
                wpe = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[1]}),
                                                            params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[1];

                // 2
                ln1w = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[2]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[2];

                // 3
                ln1b = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[3]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[3];

                // 4
                qkvw = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[4]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[4];

                // 5
                qkvb = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[5]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[5];

                // 6
                attprojw = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[6]}),
                                                                 params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[6];

                // 7
                attprojb = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[7]}),
                                                                 params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[7];

                // 8
                ln2w = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[8]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[8];

                // 9
                ln2b = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[9]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[9];

                // 10
                fcw = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[10]}),
                                                            params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[10];

                // 11
                fcb = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[11]}),
                                                            params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[11];

                // 12
                fcprojw = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[12]}),
                                                                params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[12];

                // 13
                fcprojb = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[13]}),
                                                                params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[13];

                // 14
                lnfw = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[14]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[14];

                // 15
                lnfb = std::make_unique<core::Tensor<float>>(std::vector({param_sizes[15]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[15];


                delete[] params_memory_cpu;
            }

            fcloseCheck(model_file);

            //acts_memory = NULL;
            //grads_memory = NULL;
            //m_memory = NULL;
            //v_memory = NULL;
            //grads_acts_memory = NULL;
            inputs = NULL;
            targets = NULL;
            //cpu_losses = NULL;
            batch_size = 0;
            seq_len = 0;

            // int C = model.config.channels;
            int V = vocab_size;
            int Vp = padded_vocab_size;
            int maxT = max_seq_len;
            // int L = model.config.num_layers;

            FILE *state_file = fopenCheck("../data/dataset_prepared/gpt2_124M_debug_state.bin", "rb");
            int state_header[256];
            freadCheck(state_header, sizeof(int), 256, state_file);
            if (state_header[0] != 20240327) {
                logger->error("Bad magic state file");
                exit(EXIT_FAILURE);
            }
            if (state_header[1] != 2) {
                logger->error("Bad version in state file");
                logger->error("---> HINT: try to re-run `python train_gpt2.py`");
                exit(EXIT_FAILURE);
            }
            int B = state_header[2]; // batch size, e.g. 4
            int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
            assert(0 <= T && T <= maxT);

            logger->info("[STATE] batch_size: {}, seq_len: {}", B, T);

            // inputs and expected outputs, only used for error checking
            int *x = (int *) mallocCheck(B * T * sizeof(int));
            int *y = (int *) mallocCheck(B * T * sizeof(int));
            float *expected_logits = (float *) mallocCheck(B * T * V * sizeof(float));

            // read reference information from Python
            freadCheck(x, sizeof(int), B * T, state_file);
            freadCheck(y, sizeof(int), B * T, state_file);
            freadCheck(expected_logits, sizeof(float), B * T * V, state_file);
            fcloseCheck(state_file);

            int allok = 1;

            ///TODO: Call feedforward here

            ///TODO: compare against the gold data (expected_logits)

        }

        void feedforward(sycl::queue &q, int *inputs, int *targets, int B, int T) {

            // convenience parameters
            int V = vocab_size;
            int Vp = padded_vocab_size;
            int L = num_layers;
            int NH = num_heads;
            int C = channels;

            // validate inputs, all indices must be in the range [0, V)
            for (int i = 0; i < B * T; i++) {
                assert(0 <= inputs[i] && inputs[i] < V);
                if (targets != NULL) {
                    assert(0 <= targets[i] && targets[i] < V);
                }
            }

            // allocate space for all the activations if needed (done here, lazily)
            if (!isAllocated) {
                // record the current B,T as well
                batch_size = B;
                seq_len = T;
                // and now allocate the space
                {
                    size_t Vp = padded_vocab_size;
                    size_t L = num_layers;
                    size_t NH = num_heads;
                    size_t C = channels;
                    act_sizes[0] = B * T * C; // encoded
                    act_sizes[1] = L * B * T * C; // ln1
                    act_sizes[2] = L * B * T; // ln1_mean
                    act_sizes[3] = L * B * T; // ln1_rstd
                    act_sizes[4] = L * B * T * C; // atty
                    act_sizes[5] = L * B * NH * T * T; // att
                    act_sizes[6] = L * B * T * C; // attproj
                    act_sizes[7] = L * B * T * C; // residual2
                    act_sizes[8] = L * B * T * C; // ln2
                    act_sizes[9] = L * B * T; // ln2_mean
                    act_sizes[10] = L * B * T; // ln2_rstd
                    act_sizes[11] = L * B * T * 4 * C; // fch
                    act_sizes[12] = L * B * T * 4 * C; // fch_gelu
                    act_sizes[13] = L * B * T * C; // fcproj
                    act_sizes[14] = L * B * T * C; // residual3
                    act_sizes[15] = B * T * C; // lnf
                    act_sizes[16] = B * T; // lnf_mean
                    act_sizes[17] = B * T; // lnf_rstd
                    act_sizes[18] = B * T; // losses
                    act_sizes[19] = L * B * T * 3 * C; // qkvr
                    act_sizes[20] = B * T * std::max(3 * C, std::max(NH * T, Vp)); // output / scratch
                }

                num_activations = 0;
                for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
                    num_activations += act_sizes[i];
                }

                {
                    /*
                    float** ptrs[] = {
                            &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->atty,
                            &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
                            &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
                            &acts->lnf_mean, &acts->lnf_rstd, &acts->losses, &acts->qkvr, &acts->output
                    };*/

                    // No need to copy data here, just allocate memory.
                    // 0
                    encoded = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[0]}));

                    // 1
                    ln1 = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[1]}));

                    // 2
                    ln1_mean = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[2]}));

                    // 3
                    ln1_rstd = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[3]}));

                    // 4
                    atty = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[4]}));

                    // 5
                    att = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[5]}));

                    // 6
                    attproj = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[6]}));

                    // 7
                    residual2 = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[7]}));

                    // 8
                    ln2 = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[8]}));

                    // 9
                    ln2_mean = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[9]}));

                    // 10
                    ln2_rstd = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[10]}));

                    // 11
                    fch = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[11]}));

                    // 12
                    fch_gelu = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[12]}));

                    // 13
                    fcproj = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[13]}));

                    // 14
                    residual3 = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[14]}));

                    // 15
                    lnf = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[15]}));

                    // 16
                    lnf_mean = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[16]}));

                    // 17
                    lnf_rstd = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[17]}));

                    // 18
                    losses = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[18]}));

                    // 19
                    qkvr = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[19]}));

                    // 20
                    output = std::make_unique<core::Tensor<float>>(std::vector({act_sizes[20]}));
                }
                logger->info("allocated {} MiB for activations", (num_activations * sizeof(float)) >> 20);


                // also create memory for caching inputs and targets
                this->inputs = std::make_unique<core::Tensor<int>>(std::vector<size_t>({(size_t) B * T}));
                this->targets = std::make_unique<core::Tensor<int>>(std::vector<size_t>({(size_t) B * T}));

            } else {
                // validate B,T is consistent with how we've allocated the memory before
                // in principle we could get more clever here in the future, for now this is safest
                if (B != batch_size || T != seq_len) {
                    logger->error("Model: B=%d T=%d, Desired: B=%d T=%d", batch_size, seq_len, B, T);
                    exit(EXIT_FAILURE);
                }
            }

            // copy inputs/targets to the model
            this->inputs = std::make_unique<core::Tensor<int>>(std::vector<size_t>({(size_t) B, (size_t) T}), inputs);
            this->targets = std::make_unique<core::Tensor<int>>(std::vector<size_t>({(size_t) B, (size_t) T}), targets);

            /// TODO: Check if tensor cloning is needed here.
            ///             ParameterTensors params = model->params; // for brevity
            ///             ActivationTensors acts = model->acts;

            // encoding goes into residual[0]
            kernels::EncoderKernel encoderKernel(
                    *encoded, 0,
                    *this->inputs, 0,
                    *wte, 0,
                    *wpe, 0,
                    B, T, C
            );
            encoderKernel.Launch(q, 512);

            size_t
                    offset_ln1w = 0,
                    offset_ln1b = 0,
                    offset_qkvw = 0,
                    offset_qkvb = 0,
                    offset_attprojw = 0,
                    offset_attprojb = 0,
                    offset_ln2w = 0,
                    offset_ln2b = 0,
                    offset_fcw = 0,
                    offset_fcb = 0,
                    offset_fcprojw = 0,
                    offset_fcprojb = 0;

            size_t
                    offset_ln1 = 0,
                    offset_ln1_mean = 0,
                    offset_ln1_rstd = 0,
                    offset_qkvr = 0,
                    offset_atty = 0,
                    offset_att = 0,
                    offset_attproj = 0,
                    offset_residual2 = 0,
                    offset_ln2 = 0,
                    offset_ln2_mean = 0,
                    offset_ln2_rstd = 0,
                    offset_fch = 0,
                    offset_fch_gelu = 0,
                    offset_fcproj = 0,
                    offset_residual3 = 0;

            core::Tensor<float> *residual;
            size_t residual_offset = 0;

            for (int l = 0; l < L; l++) {
                /// ------------------------------
                residual = l == 0 ? encoded.get() : residual3.get();
                if (l >= 2) {
                    // This is the original addressing:
                    //      residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
                    // So, up until l==1, residual_offset should be zero;
                    // In short, for l==2 and onwards, we can pile-up the offset each time.
                    residual_offset += B * T * C;
                }
                /// ------------------------------

                // get the pointers of the weights for this layer
                offset_ln1w += l * C;
                offset_ln1b += l * C;
                offset_qkvw += l * 3 * C * C;
                offset_qkvb += l * 3 * C;
                offset_attprojw += l * C * C;
                offset_attprojb += l * C;
                offset_ln2w += l * C;
                offset_ln2b += l * C;
                offset_fcw += l * 4 * C * C;
                offset_fcb += l * 4 * C;
                offset_fcprojw += l * C * 4 * C;
                offset_fcprojb += l * C;

                // get the pointers of the activations for this layer
                offset_ln1 += l * B * T * C;
                offset_ln1_mean += l * B * T;
                offset_ln1_rstd += l * B * T;
                offset_qkvr += l * B * T * 3 * C;
                offset_atty += l * B * T * C;
                offset_att += l * B * NH * T * T;
                offset_attproj += l * B * T * C;
                offset_residual2 += l * B * T * C;
                offset_ln2 += l * B * T * C;
                offset_ln2_mean += l * B * T;
                offset_ln2_rstd += l * B * T;
                offset_fch += l * B * T * 4 * C;
                offset_fch_gelu += l * B * T * 4 * C;
                offset_fcproj += l * B * T * C;
                offset_residual3 += l * B * T * C;

                /// ------------------------------
                // these are only needed as scratchpads for the forward pass, but
                // need not be stored for backward
                core::Tensor<float> *scratch = output.get();
                /// ------------------------------

                // now do the forward pass
                {
                    /*
                    void layernorm_forward(
                            float* out,
                            float* mean,
                            float* rstd,
                            float* inp,
                            float* weight,
                            float* bias,
                            int B, int T, int C)
                    layernorm_forward(
                            l_ln1,
                            l_ln1_mean,
                            l_ln1_rstd,
                            residual,
                            l_ln1w,
                            l_ln1b,
                            B, T, C);

                    LayerNorm(
                            core::Tensor<float> &tnOut,
                            size_t outOffset,
                            core::Tensor<float> &tnMean,
                            size_t meanOffset,
                            core::Tensor<float> &tnRstd,
                            size_t rstdOffset,
                            core::Tensor<float> &tnInp,
                            size_t inpOffset,
                            core::Tensor<float> &tnWeight,
                            size_t weightOffset,
                            core::Tensor<float> &tnBias,
                            size_t biasOffset,
                            int B, int T, int C
                    )
                    */
                    kernels::LayerNorm kernel(
                            *ln1, offset_ln1,
                            *ln1_mean, offset_ln1_mean,
                            *ln1_rstd, offset_ln1_rstd,
                            *residual, residual_offset,
                            *ln1w, offset_ln1w,
                            *ln1b, offset_ln1b,
                            B, T, C
                    );
                    kernel.Launch(q, 512);
                }

                {
                    /*
                    void matmul_forward_cublaslt(
                            float* out,
                            float* inp,
                            float* weight,
                            float* bias,
                            int B, int T, int C, int OC);
                    matmul_forward_cublaslt(
                            scratch,
                            l_ln1,
                            l_qkvw,
                            l_qkvb,
                            B, T, C, 3 * C);
                    MatmulBias(
                            core::Tensor<float> &tnOut,
                            size_t outOffset,
                            core::Tensor<float> &tnInp,
                            size_t inpOffset,
                            core::Tensor<float> &tnWeight,
                            size_t weightOffset,
                            core::Tensor<float> &tnBias,
                            size_t biasOffset,
                            int B, int T, int C, int OC,
                            bool hasBias = true
                    )
                    */
                    kernels::MatmulBias kernel(
                            *scratch, 0,
                            *ln1, offset_ln1,
                            *qkvw, offset_qkvw,
                            *qkvb, offset_qkvb,
                            B, T, C, 3 * C
                    );
                    kernel.Launch(q, 512);
                }

                {
                    /*
                    void attention_forward(
                            float* out,
                            float* qkvr,
                            float* att,
                            float* inp,
                            int B, int T, int C, int NH);

                     attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);

                     Attention(
                            core::Tensor<float> &tnOut,
                            size_t outOffset,
                            core::Tensor<float> &tnQkvr,
                            size_t qkvrOffset,
                            core::Tensor<float> &tnAtt,
                            size_t attOffset,
                            core::Tensor<float> &tnInp,
                            size_t inpOffset,
                            int B, int T, int C, int NH,
                            int blockSizeSoftMax
                    )
                    */

                    kernels::Attention kernel(
                            *atty, offset_atty,
                            *qkvr, offset_qkvr,
                            *att, offset_att,
                            *scratch, 0,
                            B, T, C, NH, 512);
                    kernel.Launch(q, 512);
                }

                {
                    /*
                    void matmul_forward_cublaslt(
                            float* out,
                            float* inp,
                            float* weight,
                            float* bias,
                            int B, int T, int C, int OC);
                    matmul_forward_cublaslt(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C)
                    MatmulBias(
                            core::Tensor<float> &tnOut,
                            size_t outOffset,
                            core::Tensor<float> &tnInp,
                            size_t inpOffset,
                            core::Tensor<float> &tnWeight,
                            size_t weightOffset,
                            core::Tensor<float> &tnBias,
                            size_t biasOffset,
                            int B, int T, int C, int OC,
                            bool hasBias = true
                    )
                    */
                    kernels::MatmulBias kernel(
                            *attproj, offset_attproj,
                            *atty, offset_atty,
                            *attprojw, offset_attprojw,
                            *attprojb, offset_attprojb,
                            B, T, C, C
                    );
                    kernel.Launch(q, 512);
                }

                {
                    /*
                    void residual_forward(float* out, float* inp1, float* inp2, int N);
                    residual_forward(l_residual2, residual, l_attproj, B * T * C);
                    Residual(
                            core::Tensor<float> &tnOutput,
                            size_t outputOffset,
                            core::Tensor<float> &tnInput1,
                            size_t input1Offset,
                            core::Tensor<float> &tnInput2,
                            size_t input2Offset,
                            int N );
                     */
                    kernels::Residual kernel(
                            *residual2, offset_residual2,
                            *residual, residual_offset,
                            *attproj, offset_attproj,
                            B * T * C);
                    kernel.Launch(q, 512);
                }

                {
                    /*
                    void layernorm_forward(
                            float* out,
                            float* mean,
                            float* rstd,
                            float* inp,
                            float* weight,
                            float* bias,
                            int B, int T, int C)
                    layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);

                    LayerNorm(
                            core::Tensor<float> &tnOut,
                            size_t outOffset,
                            core::Tensor<float> &tnMean,
                            size_t meanOffset,
                            core::Tensor<float> &tnRstd,
                            size_t rstdOffset,
                            core::Tensor<float> &tnInp,
                            size_t inpOffset,
                            core::Tensor<float> &tnWeight,
                            size_t weightOffset,
                            core::Tensor<float> &tnBias,
                            size_t biasOffset,
                            int B, int T, int C
                    )
                    */
                    kernels::LayerNorm kernel(
                            *ln2, offset_ln2,
                            *ln2_mean, offset_ln2_mean,
                            *ln2_rstd, offset_ln2_rstd,
                            *residual2, offset_residual2,
                            *ln2w, offset_ln2w,
                            *ln2b, offset_ln2b,
                            B, T, C
                    );
                    kernel.Launch(q, 512);
                }

                {
                    /*
                    void matmul_forward_cublaslt(
                            float* out,
                            float* inp,
                            float* weight,
                            float* bias,
                            int B, int T, int C, int OC);
                    matmul_forward_cublaslt(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
                    MatmulBias(
                            core::Tensor<float> &tnOut,
                            size_t outOffset,
                            core::Tensor<float> &tnInp,
                            size_t inpOffset,
                            core::Tensor<float> &tnWeight,
                            size_t weightOffset,
                            core::Tensor<float> &tnBias,
                            size_t biasOffset,
                            int B, int T, int C, int OC,
                            bool hasBias = true
                    )
                    */

                    kernels::MatmulBias kernel(
                            *fch, offset_fch,
                            *ln2, offset_ln2,
                            *fcw, offset_fcw,
                            *fcb, offset_fcb,
                            B, T, C, 4 * C
                    );
                    kernel.Launch(q, 512);
                }

                {
                    /*
                    void gelu_forward(float* out, const float* inp, int N);
                    gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
                    Gelu(
                        core::Tensor<float> &tnOutput,
                        size_t outputOffset,
                        core::Tensor<float> &tnInput,
                        size_t inputOffset,
                        int N)
                    */
                    kernels::Gelu kernel(
                            *fch_gelu, offset_fch_gelu,
                            *fch, offset_fch,
                            B * T * 4 * C
                    );
                    kernel.Launch(q, 512);
                }

                {
                    /*
                    void matmul_forward_cublaslt(
                            float* out,
                            float* inp,
                            float* weight,
                            float* bias,
                            int B, int T, int C, int OC);
                    matmul_forward_cublaslt(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
                    MatmulBias(
                            core::Tensor<float> &tnOut,
                            size_t outOffset,
                            core::Tensor<float> &tnInp,
                            size_t inpOffset,
                            core::Tensor<float> &tnWeight,
                            size_t weightOffset,
                            core::Tensor<float> &tnBias,
                            size_t biasOffset,
                            int B, int T, int C, int OC,
                            bool hasBias = true
                    )
                    */

                    kernels::MatmulBias kernel(
                            *fcproj, offset_fcproj,
                            *fch_gelu, offset_fch_gelu,
                            *fcprojw, offset_fcprojw,
                            *fcprojb, offset_fcprojb,
                            B, T, 4 * C, C
                    );
                    kernel.Launch(q, 512);
                }

                {
                    /*
                    void residual_forward(float* out, float* inp1, float* inp2, int N);
                    residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C)
                    Residual(
                            core::Tensor<float> &tnOutput,
                            size_t outputOffset,
                            core::Tensor<float> &tnInput1,
                            size_t input1Offset,
                            core::Tensor<float> &tnInput2,
                            size_t input2Offset,
                            int N );
                     */
                    kernels::Residual kernel(
                            *residual3, offset_residual3,
                            *residual2, offset_residual2,
                            *fcproj, offset_fcproj,
                            B * T * C);
                    kernel.Launch(q, 512);
                }
            }

            // last residual is in residual3
            residual = residual3.get();
            residual_offset = (L-1) * B * T * C;

            {
                /*
                void layernorm_forward(
                        float* out,
                        float* mean,
                        float* rstd,
                        float* inp,
                        float* weight,
                        float* bias,
                        int B, int T, int C)
                layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);

                LayerNorm(
                        core::Tensor<float> &tnOut,
                        size_t outOffset,
                        core::Tensor<float> &tnMean,
                        size_t meanOffset,
                        core::Tensor<float> &tnRstd,
                        size_t rstdOffset,
                        core::Tensor<float> &tnInp,
                        size_t inpOffset,
                        core::Tensor<float> &tnWeight,
                        size_t weightOffset,
                        core::Tensor<float> &tnBias,
                        size_t biasOffset,
                        int B, int T, int C
                )
                */
                kernels::LayerNorm kernel(
                        *lnf, 0,
                        *lnf_mean, 0,
                        *lnf_rstd, 0,
                        *residual, residual_offset,
                        *lnfw, 0,
                        *lnfb, 0,
                        B, T, C
                );
                kernel.Launch(q, 512);
            }

            {
                /*
                void matmul_forward_cublaslt(
                        float* out,
                        float* inp,
                        float* weight,
                        float* bias,
                        int B, int T, int C, int OC);
                matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);
                MatmulBias(
                        core::Tensor<float> &tnOut,
                        size_t outOffset,
                        core::Tensor<float> &tnInp,
                        size_t inpOffset,
                        core::Tensor<float> &tnWeight,
                        size_t weightOffset,
                        core::Tensor<float> &tnBias,
                        size_t biasOffset,
                        int B, int T, int C, int OC,
                        bool hasBias = true
                )
                */

                kernels::MatmulBias kernel(
                        *output, 0,
                        *lnf, 0,
                        *wte, 0,
                        *wte /* dummy, hasBias is set to false*/, 0,
                        B, T, C, Vp,
                        false
                );
                kernel.Launch(q, 512);
            }

            /*
            // also forward the cross-entropy loss function if we have the targets
            if (targets != NULL) {
                // fused classifier: does the forward pass and first part of the backward pass
                // we're passing dlosses = NULL, which will default them to 1.0f/(B*T), i.e. uniform loss
                fused_classifier3(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);
                // for convenience also evaluate the mean loss (TODO re-think this compute+sync point)
                // move the (B,T) losses to CPU
                cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
                float mean_loss = 0.0f;
                for (int i=0; i<B*T; i++) { mean_loss += model->cpu_losses[i]; }
                mean_loss /= B*T;
                model->mean_loss = mean_loss;

            } else {
                // if we don't have targets, we don't have loss
                model->mean_loss = -1.0f;
            }
            */

        }
    };
}
