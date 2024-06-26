//
// Created by saleh on 19/05/24.
//

#pragma once

#include "core/Tensor.h"
#include "common/utils.h"
#include "common/common.h"
#include "common/timer.h"
#include "kernels/Encoder.h"
#include "kernels/Residual.h"
#include "kernels/MatmulBias.h"
#include "kernels/Unpermute.h"
#include "kernels/Permute.h"
#include "kernels/Softmax.h"
#include "kernels/LayerNorm.h"
#include "kernels/Attention.h"
#include "kernels/Gelu.h"
#include "orig/tokenizer.h"

#include <time.h>


//#include "DataLoader.h"

#define NUM_PARAMETER_TENSORS 16
#define NUM_ACTIVATION_TENSORS 21

namespace llmsycl::model {
    class Model {
    private:
        bool disableTensorDumping = false;
        int generationBound = 0;
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
        //core::TensorPtr<int> targets; // the target tokens for the current forward pass
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
        constexpr static int GPT2_EOT = 50256;
    public:
        Model(int batch_size, int generationBound, bool disableTensorDumping) {
            this->batch_size = batch_size;
            this->generationBound = generationBound;
            this->disableTensorDumping = disableTensorDumping;
        }

        void loadCheckpoint(sycl::queue &queue, const std::string &checkpointPath) {
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
                auto *params_memory_cpu = new float[num_parameters];
                freadCheck(params_memory_cpu, sizeof(float), num_parameters, model_file);
                size_t offsetAccumulator = 0;

                // 0
                wte = std::make_unique<core::Tensor<float>>(queue,
                                                            std::vector({param_sizes[0]}),
                                                            params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[0];

                // 1
                wpe = std::make_unique<core::Tensor<float>>(queue,
                                                            std::vector({param_sizes[1]}),
                                                            params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[1];

                // 2
                ln1w = std::make_unique<core::Tensor<float>>(queue,
                                                             std::vector({param_sizes[2]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[2];

                // 3
                ln1b = std::make_unique<core::Tensor<float>>(queue,
                                                             std::vector({param_sizes[3]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[3];

                // 4
                qkvw = std::make_unique<core::Tensor<float>>(queue,
                                                             std::vector({param_sizes[4]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[4];

                // 5
                qkvb = std::make_unique<core::Tensor<float>>(queue,
                                                             std::vector({param_sizes[5]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[5];

                // 6
                attprojw = std::make_unique<core::Tensor<float>>(queue,
                                                                 std::vector({param_sizes[6]}),
                                                                 params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[6];

                // 7
                attprojb = std::make_unique<core::Tensor<float>>(queue,
                                                                 std::vector({param_sizes[7]}),
                                                                 params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[7];

                // 8
                ln2w = std::make_unique<core::Tensor<float>>(queue,
                                                             std::vector({param_sizes[8]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[8];

                // 9
                ln2b = std::make_unique<core::Tensor<float>>(queue,
                                                             std::vector({param_sizes[9]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[9];

                // 10
                fcw = std::make_unique<core::Tensor<float>>(queue,
                                                            std::vector({param_sizes[10]}),
                                                            params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[10];

                // 11
                fcb = std::make_unique<core::Tensor<float>>(queue,
                                                            std::vector({param_sizes[11]}),
                                                            params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[11];

                // 12
                fcprojw = std::make_unique<core::Tensor<float>>(queue,
                                                                std::vector({param_sizes[12]}),
                                                                params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[12];

                // 13
                fcprojb = std::make_unique<core::Tensor<float>>(queue,
                                                                std::vector({param_sizes[13]}),
                                                                params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[13];

                // 14
                lnfw = std::make_unique<core::Tensor<float>>(queue,
                                                             std::vector({param_sizes[14]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[14];

                // 15
                lnfb = std::make_unique<core::Tensor<float>>(queue,
                                                             std::vector({param_sizes[15]}),
                                                             params_memory_cpu + offsetAccumulator);
                offsetAccumulator += param_sizes[15];

                delete[] params_memory_cpu;
            }

            fcloseCheck(model_file);
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
        }

        std::vector<std::vector<sycl::event>> feedforward(sycl::queue &q, int *inputs, int *targets, int B, int T, int genIndex) {
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
                    // No need to copy data here, just allocate memory.
                    // 0
                    encoded = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[0]}));

                    // 1
                    ln1 = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[1]}));

                    // 2
                    ln1_mean = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[2]}));

                    // 3
                    ln1_rstd = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[3]}));

                    // 4
                    atty = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[4]}));

                    // 5
                    att = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[5]}));

                    // 6
                    attproj = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[6]}));

                    // 7
                    residual2 = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[7]}));

                    // 8
                    ln2 = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[8]}));

                    // 9
                    ln2_mean = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[9]}));

                    // 10
                    ln2_rstd = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[10]}));

                    // 11
                    fch = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[11]}));

                    // 12
                    fch_gelu = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[12]}));

                    // 13
                    fcproj = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[13]}));

                    // 14
                    residual3 = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[14]}));

                    // 15
                    lnf = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[15]}));

                    // 16
                    lnf_mean = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[16]}));

                    // 17
                    lnf_rstd = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[17]}));

                    // 18
                    losses = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[18]}));

                    // 19
                    qkvr = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[19]}));

                    // 20
                    output = std::make_unique<core::Tensor<float>>(q, std::vector({act_sizes[20]}));
                }
                logger->info("Allocated {} MiB for activations", (num_activations * sizeof(float)) >> 20);
                isAllocated = true;

                this->inputs = std::make_unique<core::Tensor<int>>(q, std::vector<size_t>({(size_t) B * T}), inputs);
                //this->targets = std::make_unique<core::Tensor<int>>(q, std::vector<size_t>({(size_t) B * T}), targets);

            } else {
                // validate B,T is consistent with how we've allocated the memory before
                // in principle we could get more clever here in the future, for now this is safest
                if (B != batch_size || T != seq_len) {
                    logger->error("Model: B=%d T=%d, Desired: B=%d T=%d", batch_size, seq_len, B, T);
                    exit(EXIT_FAILURE);
                }
                std::memcpy(this->inputs->getHostBuffer(), inputs, B * T * sizeof(int));
                //std::memcpy(this->targets->getHostBuffer(), targets, B * T * sizeof(int));
                this->inputs->syncBlockingH2D();
                //this->targets->syncBlockingH2D();
            }

            // copy inputs/targets to the model




            /// TODO: Check if tensor cloning is needed here.
            ///             ParameterTensors params = model->params; // for brevity
            ///             ActivationTensors acts = model->acts;


            if (!disableTensorDumping) {
                wte->syncBlockingD2H();
                wte->saveHostToNpy(0, V * C, "/tmp/c00.wte.gen" + std::to_string(genIndex) + "_uut.npy");

                wpe->syncBlockingD2H();
                wpe->saveHostToNpy(0, max_seq_len * C, "/tmp/c00.wpe.gen" + std::to_string(genIndex) + "_uut.npy");

                this->inputs->syncBlockingD2H();
                this->inputs->saveHostToNpy(0, B * T, "/tmp/c00.inp.gen" + std::to_string(genIndex) + "_uut.npy");
            }

            std::vector<std::vector<sycl::event>> vec_events;
            vec_events.push_back({});

            // encoding goes into residual[0]
            kernels::EncoderKernel encoderKernel(
                    encoded->getDeviceBuffer(),
                    this->inputs->getDeviceBuffer(),
                    wte->getDeviceBuffer(),
                    wpe->getDeviceBuffer(),
                    B, T, C
            );

            vec_events.push_back(
                    encoderKernel.Launch(q, 512, vec_events.back())
            );

            if (!disableTensorDumping) {
                encoded->syncBlockingD2H();
                encoded->saveHostToNpy(0, B * T * C, "/tmp/c01.gen" + std::to_string(genIndex) + "_uut.npy");
            }

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
                if (!disableTensorDumping) {
                    residual->syncBlockingD2H();
                    residual->saveHostToNpy(residual_offset, B * T * C,
                                            "/tmp/c02.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                            "_uut.npy");
                }
                /// ------------------------------

                // get the pointers of the weights for this layer
                offset_ln1w = l * C;
                offset_ln1b = l * C;
                offset_qkvw = l * 3 * C * C;
                offset_qkvb = l * 3 * C;
                offset_attprojw = l * C * C;
                offset_attprojb = l * C;
                offset_ln2w = l * C;
                offset_ln2b = l * C;
                offset_fcw = l * 4 * C * C;
                offset_fcb = l * 4 * C;
                offset_fcprojw = l * C * 4 * C;
                offset_fcprojb = l * C;

                // get the pointers of the activations for this layer
                offset_ln1 = l * B * T * C;
                offset_ln1_mean = l * B * T;
                offset_ln1_rstd = l * B * T;
                offset_qkvr = l * B * T * 3 * C;
                offset_atty = l * B * T * C;
                offset_att = l * B * NH * T * T;
                offset_attproj = l * B * T * C;
                offset_residual2 = l * B * T * C;
                offset_ln2 = l * B * T * C;
                offset_ln2_mean = l * B * T;
                offset_ln2_rstd = l * B * T;
                offset_fch = l * B * T * 4 * C;
                offset_fch_gelu = l * B * T * 4 * C;
                offset_fcproj = l * B * T * C;
                offset_residual3 = l * B * T * C;

                /// ------------------------------
                // these are only needed as scratchpads for the forward pass, but
                // need not be stored for backward
                core::Tensor<float> *scratch = output.get();
                /// ------------------------------

                // now do the forward pass
                {
                    kernels::LayerNorm kernel(
                            ln1->getDeviceBuffer() + offset_ln1,
                            ln1_mean->getDeviceBuffer() + offset_ln1_mean,
                            ln1_rstd->getDeviceBuffer() + offset_ln1_rstd,
                            residual->getDeviceBuffer() + residual_offset,
                            ln1w->getDeviceBuffer() + offset_ln1w,
                            ln1b->getDeviceBuffer() + offset_ln1b,
                            B, T, C
                    );
                    vec_events.push_back(kernel.Launch(q, 256, vec_events.back()));
                    if (!disableTensorDumping) {
                        ln1->syncBlockingD2H();
                        ln1->saveHostToNpy(offset_ln1, B * T * C,
                                           "/tmp/c03.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                           "_uut.npy");

                        ln1_mean->syncBlockingD2H();
                        ln1_mean->saveHostToNpy(offset_ln1_mean, B * T,
                                                "/tmp/c04.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                                "_uut.npy");

                        ln1_rstd->syncBlockingD2H();
                        ln1_rstd->saveHostToNpy(offset_ln1_rstd, B * T,
                                                "/tmp/c05.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                                "_uut.npy");
                    }
                }

                {
                    kernels::MatmulBias kernel(
                            scratch->getDeviceBuffer() + 0,
                            ln1->getDeviceBuffer() + offset_ln1,
                            qkvw->getDeviceBuffer() + offset_qkvw,
                            qkvb->getDeviceBuffer() + offset_qkvb,
                            B, T, C, 3 * C,
                            true
                    );
                    vec_events.push_back(kernel.Launch(q, 512, vec_events.back()));
                    if (!disableTensorDumping) {
                        scratch->syncBlockingD2H();
                        scratch->saveHostToNpy(0, B * T * (3 * C),
                                               "/tmp/c06.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                               "_uut.npy");
                    }
                }

                {
                    ///TODO: Fix this kernel.
                    /// atty and att are wrong.
                    /// qkvr is correct.
                    kernels::Attention kernel(
                            atty->getDeviceBuffer() + offset_atty,
                            qkvr->getDeviceBuffer() + offset_qkvr,
                            att->getDeviceBuffer() + offset_att,
                            scratch->getDeviceBuffer() + 0,
                            B, T, C, NH, 256);
                    vec_events.push_back(kernel.Launch(q, 512, vec_events.back()));

                    if (!disableTensorDumping) {
                        atty->syncBlockingD2H();
                        atty->saveHostToNpy(offset_atty, B * T * C,
                                            "/tmp/c07.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                            "_uut.npy");

                        qkvr->syncBlockingD2H();
                        qkvr->saveHostToNpy(offset_qkvr, B * T * (3 * C),
                                            "/tmp/c08.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                            "_uut.npy");

                        att->syncBlockingD2H();
                        att->saveHostToNpy(offset_att, B * NH * T * T,
                                           "/tmp/c09.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                           "_uut.npy");
                    }
                }

                {
                    kernels::MatmulBias kernel(
                            attproj->getDeviceBuffer() + offset_attproj,
                            atty->getDeviceBuffer() + offset_atty,
                            attprojw->getDeviceBuffer() + offset_attprojw,
                            attprojb->getDeviceBuffer() + offset_attprojb,
                            B, T, C, C
                    );
                    vec_events.push_back(kernel.Launch(q, 512, vec_events.back()));
                    if (!disableTensorDumping) {
                        attproj->syncBlockingD2H();
                        attproj->saveHostToNpy(offset_attproj, B * T * C,
                                               "/tmp/c10.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                               "_uut.npy");
                    }
                }

                {
                    kernels::Residual kernel(
                            residual2->getDeviceBuffer() + offset_residual2,
                            residual->getDeviceBuffer() + residual_offset,
                            attproj->getDeviceBuffer() + offset_attproj,
                            B * T * C);
                    vec_events.push_back(
                            kernel.Launch(q, 512, vec_events.back())
                    );
                    if (!disableTensorDumping) {
                        residual2->syncBlockingD2H();
                        residual2->saveHostToNpy(offset_residual2, B * T * C,
                                                 "/tmp/c11.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                                 "_uut.npy");
                    }
                }

                {
                    kernels::LayerNorm kernel(
                            ln2->getDeviceBuffer() + offset_ln2,
                            ln2_mean->getDeviceBuffer() + offset_ln2_mean,
                            ln2_rstd->getDeviceBuffer() + offset_ln2_rstd,
                            residual2->getDeviceBuffer() + offset_residual2,
                            ln2w->getDeviceBuffer() + offset_ln2w,
                            ln2b->getDeviceBuffer() + offset_ln2b,
                            B, T, C
                    );
                    vec_events.push_back(
                            kernel.Launch(q, 256, vec_events.back())
                    );
                    if (!disableTensorDumping) {
                        ln2->syncBlockingD2H();
                        ln2_mean->syncBlockingD2H();
                        ln2_rstd->syncBlockingD2H();
                        ln2->saveHostToNpy(offset_ln2, B * T * C,
                                           "/tmp/c12.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                           "_uut.npy");
                        ln2_mean->saveHostToNpy(offset_ln2_mean, B * T,
                                                "/tmp/c13.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                                "_uut.npy");
                        ln2_rstd->saveHostToNpy(offset_ln2_rstd, B * T,
                                                "/tmp/c14.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                                "_uut.npy");
                    }
                }

                {
                    kernels::MatmulBias kernel(
                            fch->getDeviceBuffer() + offset_fch,
                            ln2->getDeviceBuffer() + offset_ln2,
                            fcw->getDeviceBuffer() + offset_fcw,
                            fcb->getDeviceBuffer() + offset_fcb,
                            B, T, C, 4 * C
                    );
                    vec_events.push_back(
                            kernel.Launch(q, 512, vec_events.back())
                    );
                    if (!disableTensorDumping) {
                        fch->syncBlockingD2H();
                        fch->saveHostToNpy(offset_fch, B * T * 4 * C,
                                           "/tmp/c15.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                           "_uut.npy");
                    }
                }

                {
                    kernels::Gelu kernel(
                            fch_gelu->getDeviceBuffer() + offset_fch_gelu,
                            fch->getDeviceBuffer() + offset_fch,
                            B * T * 4 * C
                    );
                    vec_events.push_back(
                            kernel.Launch(q, 512, vec_events.back())
                    );
                    if (!disableTensorDumping) {
                        fch_gelu->syncBlockingD2H();
                        fch_gelu->saveHostToNpy(offset_fch_gelu, B * T * 4 * C,
                                                "/tmp/c16.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                                "_uut.npy");
                    }
                }

                {
                    kernels::MatmulBias kernel(
                            fcproj->getDeviceBuffer() + offset_fcproj,
                            fch_gelu->getDeviceBuffer() + offset_fch_gelu,
                            fcprojw->getDeviceBuffer() + offset_fcprojw,
                            fcprojb->getDeviceBuffer() + offset_fcprojb,
                            B, T, 4 * C, C
                    );
                    vec_events.push_back(
                            kernel.Launch(q, 512, vec_events.back())
                    );
                    if (!disableTensorDumping) {
                        fcproj->syncBlockingD2H();
                        fcproj->saveHostToNpy(offset_fcproj, B * T * C,
                                              "/tmp/c17.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                              "_uut.npy");
                    }
                }

                {
                    kernels::Residual kernel(
                            residual3->getDeviceBuffer() + offset_residual3,
                            residual2->getDeviceBuffer() + offset_residual2,
                            fcproj->getDeviceBuffer() + offset_fcproj,
                            B * T * C);
                    vec_events.push_back(
                            kernel.Launch(q, 512, vec_events.back())
                    );
                    if (!disableTensorDumping) {
                        residual3->syncBlockingD2H();
                        residual3->saveHostToNpy(offset_residual3, B * T * C,
                                                 "/tmp/c18.l" + std::to_string(l) + ".gen" + std::to_string(genIndex) +
                                                 "_uut.npy");
                    }
                }
            }

            // last residual is in residual3
            residual = residual3.get();
            residual_offset = (L - 1) * B * T * C;
            if (!disableTensorDumping) {
                residual->syncBlockingD2H();
                residual->saveHostToNpy(residual_offset, B * T * C,
                                        "/tmp/c19.gen" + std::to_string(genIndex) + "_uut.npy");
            }
            {
                kernels::LayerNorm kernel(
                        lnf->getDeviceBuffer() + 0,
                        lnf_mean->getDeviceBuffer() + 0,
                        lnf_rstd->getDeviceBuffer() + 0,
                        residual->getDeviceBuffer() + residual_offset,
                        lnfw->getDeviceBuffer() + 0,
                        lnfb->getDeviceBuffer() + 0,
                        B, T, C
                );
                vec_events.push_back(
                        kernel.Launch(q, 256, vec_events.back())
                );
                if (!disableTensorDumping) {
                    lnf->syncBlockingD2H();
                    lnf->saveHostToNpy(0, B * T * C, "/tmp/c20.gen" + std::to_string(genIndex) + "_uut.npy");
                    lnf_mean->syncBlockingD2H();
                    lnf_mean->saveHostToNpy(0, B * T, "/tmp/c21.gen" + std::to_string(genIndex) + "_uut.npy");
                    lnf_rstd->syncBlockingD2H();
                    lnf_rstd->saveHostToNpy(0, B * T, "/tmp/c22.gen" + std::to_string(genIndex) + "_uut.npy");
                }
            }

            {
                kernels::MatmulBias kernel(
                        output->getDeviceBuffer() + 0,
                        lnf->getDeviceBuffer() + 0,
                        wte->getDeviceBuffer() + 0,
                        wte->getDeviceBuffer() + 0, //dummy, `hasBias` is set to false
                        B, T, C, Vp,
                        false
                );
                vec_events.push_back(
                        kernel.Launch(q, 512, vec_events.back())
                );
                if (!disableTensorDumping) {
                    output->syncBlockingD2H();
                    output->saveHostToNpy(0, B * T * Vp, "/tmp/c23.gen" + std::to_string(genIndex) + "_uut.npy");
                }
            }
            return vec_events;
        }

        unsigned int random_u32(unsigned long long *state) {
            // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
            *state ^= *state >> 12;
            *state ^= *state << 25;
            *state ^= *state >> 27;
            return (*state * 0x2545F4914F6CDD1Dull) >> 32;
        }

        int sample_softmax(const float *logits, int n, float coin) {
            // sample index from logits (converted to probabilities using softmax)
            // coin is a random number in [0, 1), usually from random_f32()
            double norm = 0;
            for (int i = 0; i < n; i++) {
                norm += std::exp(logits[i]);
            }
            // instead of dividing all exp(logits), we can just multiply coin.
            coin *= norm;
            float cdf = 0.0f;
            for (int i = 0; i < n; i++) {
                cdf += std::exp(logits[i]);
                if (coin < cdf) {
                    return i;
                }
            }
            return n - 1; // in case of rounding errors
        }

        std::map<int, std::vector<std::vector<sycl::event>>> inference(sycl::queue &sycl_queue) {
            struct timespec start, end;
            std::map<int, std::vector<std::vector<sycl::event>>> events_per_gen;
            const char *train_data_pattern = "../data/dataset_prepared/tiny_shakespeare_train.bin";
            const char *val_data_pattern = "../data/dataset_prepared/tiny_shakespeare_val.bin";
            const char *output_log_file = NULL;
            int B = 1; // batch size
            int T = 1024; // sequence length max
            float learning_rate = 3e-4f;
            int val_loss_every = 20; // every how many steps do we eval validation loss?
            int val_max_steps = 20; // how many batches max do we eval for validation loss?
            int sample_every = 20; // every how many steps to do inference?
            int genT = generationBound; // number of steps of inference we will do

            logger->info("+-----------------------+----------------------------------------------------+\n");
            logger->info("| Parameter             | Value                                              |\n");
            logger->info("+-----------------------+----------------------------------------------------+\n");
            logger->info("| train data pattern    | {} |", train_data_pattern);
            logger->info("| val data pattern      | {} |", val_data_pattern);
            logger->info("| output log file       | {} |", output_log_file == NULL ? "NULL" : output_log_file);
            logger->info("| batch size B          | {} |", B);
            logger->info("| sequence length T     | {} |", T);
            logger->info("| learning rate         | {} |", learning_rate);
            logger->info("| val_loss_every        | {} |", val_loss_every);
            logger->info("| val_max_steps         | {} |", val_max_steps);
            logger->info("| sample_every          | {} |", sample_every);
            logger->info("| genT                  | {} |", genT);
            logger->info("+-----------------------+----------------------------------------------------+\n");

            // build the GPT-2 model from a checkpoint
            loadCheckpoint(sycl_queue, "../data/dataset_prepared/gpt2_124M.bin");
            logger->info("| max_sequence_length T | {} |", max_seq_len);
            logger->info("| vocab_size V          | {} |", vocab_size);
            logger->info("| padded_vocab_size Vp  | {} |", padded_vocab_size);
            logger->info("| num_layers L          | {} |", num_layers);
            logger->info("| num_heads NH          | {} |", num_heads);
            logger->info("| channels C            | {} |", channels);
            logger->info("| num_parameters        | {} |", num_parameters);
            logger->info("+-----------------------+----------------------------------------------------+\n");

            // build the Tokenizer
            Tokenizer tokenizer;
            tokenizer_init(&tokenizer, "../data/dataset_prepared/gpt2_tokenizer.bin");

            // some memory for generating samples from the model
            unsigned long long rng_state = 1337;
            int *gen_tokens = (int *) mallocCheck(B * T * sizeof(int));

            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for (int i = 0; i < B * T; ++i) {
                gen_tokens[i] = GPT2_EOT;
            }

            // now sample from the model autoregressively
            clock_gettime(CLOCK_MONOTONIC, &start);
            logger->info("generating:\n---\n");

            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                events_per_gen[t-1] = feedforward(sycl_queue, gen_tokens, NULL, B, T, t - 1);

                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
                // get the V-dimensional vector probs[0, t-1, :]
                sycl_queue.wait();


                // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
                output->syncBlockingD2H();


                // We have to read only `vocab_size` words
                auto accHostLogits = output->getHostBuffer() + (t - 1) * padded_vocab_size;

                float coin = (random_u32(&rng_state) >> 8) / 16777216.0f;
                //printf("\nCoin: %f, done running feedforward for genIndex: %d\n", coin, t-1);
                int next_token = sample_softmax(accHostLogits, vocab_size, coin);
                gen_tokens[t] = next_token;

                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char *token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);

                //std::exit(44);
            }

            clock_gettime(CLOCK_MONOTONIC, &end);
            logger->info("Total time taken for inference on host (only the genT loop): {} ms.", ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1000);


            return events_per_gen;
        }

    };
}
