//
// Created by saleh on 13/05/24.
//

#include <iostream>
#include <cassert>

#include "argparse/argparse.hpp"
#include "common/common.h"
#include "model/gpt2_v1/Model.h"

int main(int argc, char *argv[]) {
    initLogger();

    argparse::ArgumentParser program("LLM_SYCL");
    program.add_argument("-b", "--batch").default_value(1).store_into(globalBatchsize);
    program.add_argument("-g", "--gen").default_value(64).store_into(globalGeneration);
    program.add_argument("-x", "--disabledumps").default_value(false).store_into(globalDisableTensorDumping);
    program.add_argument("-d", "--datadir").default_value("../").store_into(globalDirData);
    program.add_argument("-l", "--logdir").default_value("/tmp/").store_into(globalDirLog);
    program.add_argument("-s", "--silent").flag().store_into(globalIsSilent);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        logger->error("Bad Arguments.");
        logger->error(err.what());
        return 1;
    }

    logger->debug("Data Directory: {}", globalDirLog);
    logger->debug("Log Directory: {}", globalDirLog);
    logger->debug("IsSilent: {}", globalIsSilent);
    logger->debug("BatchSize: {}", globalBatchsize);
    logger->info("Sycl version of llm.c for HPC course 2024 (Prof. B. Cosenza).");


    auto asycExceptionHandler = [](sycl::exception_list e_list) {
        for (std::exception_ptr const &e: e_list) {
            try {
                std::rethrow_exception(e);
            }
            catch (std::exception const &e) {
                logger->error("Failure: {}", e.what());
                std::terminate();
            }
        }
    };
    auto sycl_queue = sycl::queue(sycl::gpu_selector_v, asycExceptionHandler,
                                  {
                                          sycl::property::queue::enable_profiling(),
                                          //sycl::property::queue::in_order()
                                  });
    logger->info("SYCL queue initialized.");
    logger->info("Device Name: {}", sycl_queue.get_device().get_info<sycl::info::device::name>());
    logger->info("Global Memory: {}", sycl_queue.get_device().get_info<sycl::info::device::global_mem_size>());
    logger->info("Local Memory: {}", sycl_queue.get_device().get_info<sycl::info::device::local_mem_size>());
    logger->info("CUs: {}", sycl_queue.get_device().get_info<sycl::info::device::max_compute_units>());


    // Create a GPT2 model
    llmsycl::model::Model gpt2(globalBatchsize, globalGeneration, globalDisableTensorDumping);
    gpt2.inference(sycl_queue);

    logger->info("Finished inference.");
}