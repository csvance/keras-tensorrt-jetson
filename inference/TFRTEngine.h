/*
 * TFRTEngine.h
 *
 *  Created on: May 22, 2018
 *      Author: cvance
 */

#ifndef TFRTENGINE_H_
#define TFRTENGINE_H_

#include <string>
#include <vector>

#include "common.h"

class TFRTEngine {
public:
	TFRTEngine();
	virtual ~TFRTEngine();

	void addInput(string, nvinfer1::DimsCHW, size_t);
	void addOutput(string, size_t);
	bool loadUff(const char*, size_t, nvinfer1::DataType);

	std::vector<std::vector<void*>> predict(std::vector<std::vector<void*>>);

	string engineSummary();

private:
	ICudaEngine* engine;
	IExecutionContext* context;
	IUffParser* parser;
	Logger logger;

	int maxBatchSize;
	int numBindings;

	vector<size_t> networkInputs;
	vector<size_t> networkOutputs;

	std::vector<void*> GPU_Buffers;

	void allocGPUBuffer();
	void freeGPUBuffer();
};

#endif /* TFRTENGINE_H_ */
