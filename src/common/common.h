//
// Created by saleh on 16/05/24.
//

#pragma once

#include <string>
#include "spdlog/spdlog.h"

extern spdlog::logger *logger;
extern int globalBatchsize;
extern int globalGeneration;
extern bool globalDisableTensorDumping;
extern bool globalIsSilent;
extern bool globalInOrder;
extern std::string globalDirData;
extern std::string globalDirLog;

extern void initLogger();