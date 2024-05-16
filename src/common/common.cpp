//
// Created by saleh on 16/05/24.
//

#include "common.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <fstream>

spdlog::logger *logger;
int globalBatchsize;
bool globalIsSilent;
std::string globalDirData;
std::string globalDirLog;

void initLogger() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::trace);
    console_sink->set_pattern("[%H:%M:%S.%e][%^%l%$] %v");

    auto file_sink1 = std::make_shared<spdlog::sinks::basic_file_sink_mt>(globalDirLog + "LogTrace.log", true);
    file_sink1->set_level(spdlog::level::trace);
    file_sink1->set_pattern("[%H:%M:%S.%f][%^%l%$] %v");


    auto file_sink3 = std::make_shared<spdlog::sinks::basic_file_sink_mt>(globalDirLog + "LogWarn.log", true);
    file_sink3->set_level(spdlog::level::warn);
    file_sink3->set_pattern("[%H:%M:%S.%f][%^%l%$] %v");

    logger = new spdlog::logger("LLM_SYCL", {console_sink, file_sink1, file_sink3});

    logger->set_level(
            LOG==0 ? spdlog::level::level_enum::trace :
            LOG==1 ? spdlog::level::level_enum::debug :
            LOG==2 ? spdlog::level::level_enum::info :
            LOG==3 ? spdlog::level::level_enum::warn :
            LOG==4 ? spdlog::level::level_enum::err :
            LOG==5 ? spdlog::level::level_enum::critical :
            LOG==6 ? spdlog::level::level_enum::off :
            spdlog::level::level_enum::trace);

    logger->info("Logging level: {}", LOG);
    spdlog::flush_every(std::chrono::seconds(30));
}