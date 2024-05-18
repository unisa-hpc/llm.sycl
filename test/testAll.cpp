#include "gtest/gtest.h"
#include "common/common.h"



//
// Created by saleh on 13/05/24.
//

#include <iostream>
#include <cassert>

#include "argparse/argparse.hpp"
#include "common/common.h"

using namespace std;

int main(int argc, const char *argv[]) {
    initLogger();

    argparse::ArgumentParser program("LLM_SYCL Tests");
    program.add_argument("-b", "--batch").default_value(1).store_into(globalBatchsize);
    program.add_argument("-d", "--datadir").default_value("../").store_into(globalDirData);
    program.add_argument("-l", "--logdir").default_value("/tmp/").store_into(globalDirLog);
    program.add_argument("-s", "--silent").flag().store_into(globalIsSilent);
    program.add_argument("--gtest_color");
    program.add_argument("--gtest_filter");

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
    logger->trace("Commencing to run all the unit tests...");

    ::testing::InitGoogleTest(&argc, const_cast<char **>(argv));
    auto exitCode = RUN_ALL_TESTS();

    logger->trace("Closing.");
    return exitCode;
}