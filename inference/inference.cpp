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

#include "NvInfer.h"

#include "TFRTEngine.h"

#define NUM_SAMPLES 10
#define BATCH_SIZE 16*16

int main(int argc, char** argv) {
	TFRTEngine engine = TFRTEngine();
	engine.addInput("input_1", DimsCHW(1, 30, 40), sizeof(float));
	engine.addInput("input_2", DimsCHW(1, 30, 40), sizeof(float));
	engine.addOutput("dense_1/BiasAdd", sizeof(float));

	engine.loadUff("./graph.uff", (size_t) BATCH_SIZE,
			nvinfer1::DataType::kFLOAT);
	std::cout << engine.engineSummary() << std::endl;

	/* Allocate memory for predictions */
	vector<vector<void*>> batch(BATCH_SIZE);
	for (int b; b < BATCH_SIZE; b++) {

		//Inputs
		batch[b].push_back(new unsigned char[1 * 30 * 40 * 4]);
		batch[b].push_back(new unsigned char[1 * 30 * 40 * 4]);
	}

	for (;;) {
		int totalMs = 0;

		for (int i = 0; i < NUM_SAMPLES; i++) {
			auto t_start = std::chrono::high_resolution_clock::now();

			vector<vector<void*>> batchOutputs = engine.predict(batch);
			for (int b = 0; b < batchOutputs.size(); b++)
				delete batchOutputs[b][0];

			auto t_end = std::chrono::high_resolution_clock::now();
			int ms =
					std::chrono::duration<float, std::milli>(t_end - t_start).count();
			totalMs += ms;

		}

		totalMs /= NUM_SAMPLES;
		std::cout << "Average over " << NUM_SAMPLES << " runs is " << totalMs
				<< " ms." << std::endl;

	}
}
