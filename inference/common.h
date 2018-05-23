#ifndef __COMMON_H__
#define __COMMON_H__

#include <iostream>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;

inline int64_t volume(const Dims& d) {
	int64_t v = 1;
	for (int64_t i = 0; i < d.nbDims; i++)
		v *= d.d[i];
	return v;
}

void* safeCudaMalloc(size_t);

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_mnist: " + std::string(message);            \
        logger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

// Logger for GIE info/warning/errors
class Logger: public nvinfer1::ILogger {
public:
	void log(nvinfer1::ILogger::Severity, const char*) override;
};

#endif
