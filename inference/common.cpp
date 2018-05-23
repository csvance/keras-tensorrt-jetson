#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cassert>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"

#include "common.h"

using namespace nvuffparser;
using namespace nvinfer1;

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) {
	// suppress info-level messages
	if (severity == Severity::kINFO)
		return;

	switch (severity) {
	case Severity::kINTERNAL_ERROR:
		std::cerr << "INTERNAL_ERROR: ";
		break;
	case Severity::kERROR:
		std::cerr << "ERROR: ";
		break;
	case Severity::kWARNING:
		std::cerr << "WARNING: ";
		break;
	case Severity::kINFO:
		std::cerr << "INFO: ";
		break;
	default:
		std::cerr << "UNKNOWN: ";
		break;
	}
	std::cerr << msg << std::endl;
}

void* safeCudaMalloc(size_t memSize) {
	void* deviceMem;
	CHECK(cudaMalloc(&deviceMem, memSize));
	if (deviceMem == nullptr) {
		std::cerr << "Out of memory" << std::endl;
		exit(1);
	}
	return deviceMem;
}
