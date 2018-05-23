/*
 * TFRTEngine.cpp
 *
 *  Created on: May 22, 2018
 *      Author: cvance
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <string>
#include <sstream>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;

#include "TFRTEngine.h"

TFRTEngine::TFRTEngine() {
	parser = createUffParser();
	numBindings = 0;
	maxBatchSize = 0;
	context = NULL;
	engine = NULL;
}

TFRTEngine::~TFRTEngine() {
	/* Clean up */
	freeGPUBuffer();
	context->destroy();
	engine->destroy();
	shutdownProtobufLibrary();
}

void TFRTEngine::addInput(string layer, nvinfer1::DimsCHW dims,
		size_t eleSize) {
	/*
	 Register tensorflow input
	 Even if channel index is last in the data, put it first for TensorRT
	 This network inputs are defined in Keras as follows:
	 Input(shape=(Y, X, C))
	 Where Y = 30, X = 40, C = 1
	 */
	parser->registerInput(layer.c_str(), dims);

	/* Save the size for inferences */
	networkInputs.push_back(volume(dims) * eleSize);
}
void TFRTEngine::addOutput(string layer, size_t eleSize) {
	/*
	 Name of last operation of last non optimizer layer found with
	 `convert_to_uff.py tensorflow --input-file graph.pb -l`
	 A dimension is not neccisary
	 */
	parser->registerOutput(layer.c_str());

	/* Save the size for inferences */
	networkOutputs.push_back(eleSize);
}

void TFRTEngine::allocGPUBuffer() {
	int stepSize = networkInputs.size() + networkOutputs.size();

	GPU_Buffers = vector<void*>(maxBatchSize * stepSize);

	/* Allocate GPU Input memory and move input data to it for each batch*/
	for (int b = 0; b < maxBatchSize; b++) {

		/* Allocate GPU Input memory */
		int bindingIdx = 0;
		for (int i = 0; i < networkInputs.size(); i++) {
			size_t inputSize = networkInputs[i];

			GPU_Buffers[bindingIdx + b * stepSize] = safeCudaMalloc(inputSize);
			bindingIdx++;
		}

		/* Allocate GPU Output memory */
		for (int i = 0; i < networkOutputs.size(); i++) {
			size_t outputSize = networkOutputs[i];

			GPU_Buffers[bindingIdx + b * stepSize] = safeCudaMalloc(outputSize);
			bindingIdx++;
		}

	}
}

void TFRTEngine::freeGPUBuffer() {
	/* Move outputs back to host memory for each batch */
	for (int b = 0; b < GPU_Buffers.size(); b++)
		CHECK(cudaFree(GPU_Buffers[b]));
}

bool TFRTEngine::loadUff(const char* uffFile, size_t maximumBatchSize,
		nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT) {

	maxBatchSize = maximumBatchSize;

	IBuilder* builder = createInferBuilder(logger);

	INetworkDefinition* network = builder->createNetwork();

	if (dataType == DataType::kFLOAT) {
		if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
			RETURN_AND_LOG(false, ERROR, "Fail to parse");
	} else if (dataType == DataType::kHALF) {
		if (!parser->parse(uffFile, *network, nvinfer1::DataType::kINT8))
			RETURN_AND_LOG(false, ERROR, "Fail to parse");
		builder->setHalf2Mode(true);
	} else if (dataType == DataType::kINT8) {
		if (!parser->parse(uffFile, *network, nvinfer1::DataType::kINT8))
			RETURN_AND_LOG(false, ERROR, "Fail to parse");
		builder->setInt8Mode(true);
	}

	builder->setMaxBatchSize(maxBatchSize);

	/* TODO: What is this for? */
	builder->setMaxWorkspaceSize((1 << 30));

	engine = builder->buildCudaEngine(*network);
	if (!engine)
		RETURN_AND_LOG(false, ERROR, "Unable to create engine");

	/* we can clean the network and the parser */
	network->destroy();
	builder->destroy();

	/* we need to keep the memory created by the parser */
	parser->destroy();

	context = engine->createExecutionContext();

	numBindings = engine->getNbBindings();

	/* Allocate Buffers */
	allocGPUBuffer();

}

std::vector<std::vector<void*>> TFRTEngine::predict(
		std::vector<std::vector<void*>> batchInputs) {

	assert(batchInputs.size() <= maxBatchSize);

	int batchCount = batchInputs.size();
	int stepSize = networkInputs.size() + networkOutputs.size();

	/* Copy batch to GPU */
	for (int b = 0; b < batchCount; b++) {

		int bindingIdx = 0;
		for (int i = 0; i < networkInputs.size(); i++) {
			size_t inputSize = networkInputs[i];

			CHECK(
					cudaMemcpy(GPU_Buffers[bindingIdx + b * stepSize],
							batchInputs[b][i], inputSize,
							cudaMemcpyHostToDevice));
			bindingIdx++;
		}

	}

	/* Do the inference */
	assert(context->execute(batchCount, &GPU_Buffers[0]) == true);

	std::vector<std::vector<void*>> Output_Buffers(batchCount);

	/* Move outputs back to host memory for each batch */
	for (int b = 0; b < batchCount; b++) {

		int bindingIdx = batchInputs[b].size();
		for (int i = 0; i < networkOutputs.size(); i++) {
			size_t outputSize = networkOutputs[i];

			/* Allocate a host buffer for the network output */
			Output_Buffers[b].push_back(new unsigned char[outputSize]);

			CHECK(
					cudaMemcpy(Output_Buffers[b][i],
							GPU_Buffers[bindingIdx + b * stepSize], outputSize,
							cudaMemcpyDeviceToHost));

			bindingIdx++;
		}

	}

	return Output_Buffers;

}

string TFRTEngine::engineSummary() {

	std::stringstream summary;

	for (int i = 0; i < numBindings; ++i) {

		Dims dims = engine->getBindingDimensions(i);
		DataType dtype = engine->getBindingDataType(i);

		summary << "--Binding " << i << "--" << std::endl;
		if (engine->bindingIsInput(i))
			summary << "Type: Input";
		else
			summary << "Type: Output";
		summary << " DataType: ";
		if (dtype == DataType::kFLOAT)
			summary << "kFLOAT";
		else if (dtype == DataType::kHALF)
			summary << "kHALF";
		else if (dtype == DataType::kINT8)
			summary << "kINT8";

		summary << " Dims: (";
		for (int j = 0; j < dims.nbDims; j++)
			summary << dims.d[j] << ",";
		summary << ")" << std::endl;

	}
	return summary.str();

}
