//
// Created by saleh on 30/05/24.
//

#pragma once


#include <cstdio>
#include <common/utils.h>
#include <common/common.h>
#include <core/Tensor.h>

namespace llmsycl::model {
    class DataLoader {
    private:
        // hyperparameters
        int B, T;
        // input handling and its state
        FILE* tokens_file;
        long file_size;
        long current_position;

        // All of these are in host memory
        int* batch;
        int* inputs;
        int* targets;

        // convenience variables
        long num_batches;

        public:
        DataLoader(const char* filename, int B, int T) {
            this->B = B;
            this->T = T;

            // open the input file for reading
            tokens_file = fopenCheck(filename, "rb");

            // determine the file size
            fseekCheck(tokens_file, 0, SEEK_END);
            file_size = ftell(tokens_file);
            fseekCheck(tokens_file, 0, SEEK_SET);
            if (file_size < (B * T + 1) * sizeof(int)) {
                logger->error("Error: file size is too small for the batch size and sequence length");
                exit(EXIT_FAILURE);
            }
            current_position = 0; // start at the beginning

            //cudaMallocHost((void**)&loader->batch, (B * T + 1) * sizeof(int));
            batch = new int[B * T + 1];
            inputs = batch;
            targets = batch + 1;
            num_batches = file_size / (B * T * sizeof(int));
        }

        void reset() {
            current_position = 0;
        }

        void nextBatch() {
            // if we are at the end of the file, loop back to the beginning
            if (current_position + (B*T+1) * sizeof(int) > file_size) {
                current_position = 0;
            }
            // read the B*T+1 integers from the file into batch
            fseekCheck(tokens_file, current_position, SEEK_SET);
            freadCheck(batch, sizeof(int), B*T+1, tokens_file);
            // advance the current position by B*T integers
            current_position += B*T * sizeof(int);
        }

        ~DataLoader() {
            fclose(tokens_file);
            delete[] batch;
        }

    };

}