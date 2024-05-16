//
// Created by saleh on 13/05/24.
//

#include <iostream>
#include <cassert>

#include "argparse/argparse.hpp"
#include "common/common.h"

int main(int argc, char *argv[]) {
    initLogger();

    argparse::ArgumentParser program("LLM_SYCL");
    program.add_argument("-b", "--batch").implicit_value(true).default_value(1);
    program.add_argument("-d", "--datadir").implicit_value(true).default_value("../");
    program.add_argument("-l", "--logdir").implicit_value(true).default_value("/tmp/");
    program.add_argument("-s", "--silent").flag();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        logger->error("Bad Arguments.");
        logger->error(err.what());
        return 1;
    }

    globalBatchsize = program.get<int>("-b");
    globalDirData = program.get<std::string>("-d");
    globalDirLog = program.get<std::string>("-l");
    globalIsSilent = program.get<bool>("-s");
    logger->debug("Data Directory: {}", globalDirLog);
    logger->debug("Log Directory: {}", globalDirLog);
    logger->debug("IsSilent: {}", globalIsSilent);
    logger->debug("BatchSize: {}", globalBatchsize);

    logger->info("Sycl version of llm.c for HPC course 2024 (Prof. B. Cosenza).");
}